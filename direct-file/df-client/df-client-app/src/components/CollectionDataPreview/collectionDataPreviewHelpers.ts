/* eslint-disable df-rules/no-factgraph-save */
import {
  CollectionFactory,
  ConcretePath,
  DollarFactory,
  EinFactory,
  FactGraph,
  scalaListToJsArray,
  TinFactory,
} from '@irs/js-factgraph-scala';
import { Path } from '../../flow/Path.js';
import { AbsolutePath } from '../../fact-dictionary/Path.js';

export enum CollectionDataPreviewVariant {
  FORM_W2 = `/formW2s`,
  INTEREST_REPORT = `/interestReports`,
}

type CollectionPayload = { id: string; box1: string | null; ein: string | null; payerTin: string | null };

export const getDataImportCollectionKey = (collectionContext: ConcretePath) => {
  switch (collectionContext) {
    case CollectionDataPreviewVariant.FORM_W2:
      return `w2s`;
    case CollectionDataPreviewVariant.INTEREST_REPORT:
      return `interestIncome`;
    default:
      throw new Error(`No data import collection key found for ${collectionContext}`);
  }
};

const getOfferedCollectionPath = (collectionContext: ConcretePath) => {
  switch (collectionContext) {
    case CollectionDataPreviewVariant.FORM_W2:
      return `/offeredFormW2s`;
    case CollectionDataPreviewVariant.INTEREST_REPORT:
      return `/offeredInterestReports`;
    default:
      throw new Error(`No data import offered collection path found for ${collectionContext}`);
  }
};

export const setOfferedMetadataFacts = (
  collectionContext: ConcretePath,
  collectionArray: CollectionPayload[],
  factGraph: FactGraph
) => {
  const offeredCollectionPath = getOfferedCollectionPath(collectionContext);
  const offeredCollectionFactPath = offeredCollectionPath as ConcretePath;
  const collectionResult = factGraph.get(offeredCollectionFactPath);
  const collectionItems: string[] = collectionResult.complete
    ? scalaListToJsArray(collectionResult.get.getItemsAsStrings())
    : [];
  const newItemIds = collectionArray.map((c) => c.id);
  if (!collectionItems.some((item) => newItemIds.includes(item))) {
    const newCollectionItems = [...collectionItems, ...newItemIds];
    // Initialize the new collection.
    factGraph.set(offeredCollectionFactPath, CollectionFactory(newCollectionItems));
    factGraph.save();
    // Set the datetime for when the collection was offered.
    collectionArray.forEach((offered) => {
      const currentDatetime = new Date().toLocaleString();
      factGraph.set(Path.concretePath(`${offeredCollectionPath}/*/offeredAt`, offered.id), currentDatetime);
      if (offeredCollectionPath === `/offeredFormW2s`) {
        if (offered.ein) {
          factGraph.set(Path.concretePath(`${offeredCollectionPath}/*/ein`, offered.id), EinFactory(offered.ein).right);
        }
      }
      if (offeredCollectionPath === `/offeredInterestReports`) {
        if (offered.payerTin) {
          factGraph.set(
            Path.concretePath(`${offeredCollectionPath}/*/tin`, offered.id),
            TinFactory(offered.payerTin).right
          );
        }
        if (offered.box1) {
          factGraph.set(
            Path.concretePath(`${offeredCollectionPath}/*/incomeValue`, offered.id),
            DollarFactory(offered.box1).right
          );
        }
      }
    });
    factGraph.save();
  }
};

export const setImportedMetaDataFacts = (profileKey: string, factGraph: FactGraph, item: string) => {
  // Set the datetime for when the W2 was imported.
  const currentDatetime = new Date().toLocaleString();
  const importedAtPath = `${profileKey}/*/importedAt`;
  factGraph.set(Path.concretePath(importedAtPath as AbsolutePath, item), currentDatetime);
};
