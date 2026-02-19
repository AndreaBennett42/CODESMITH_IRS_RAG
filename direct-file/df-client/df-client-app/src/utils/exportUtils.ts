import { FactGraph, scalaListToJsArray } from '@irs/js-factgraph-scala';
import { exportToStateFacts } from '../fact-dictionary/generated/exportToStateFacts.js';
import { Path } from '../flow/Path.js';
import { Path as FDPath } from '../fact-dictionary/Path.js';
import { paths } from '../fact-dictionary/generated/paths.js';
// eslint-disable-next-line max-len
import { SENSITIVE_EXPORTABLE_FACTS } from '../screens/AuthorizeStateScreen/TransferInfoModal/TransferredDataList/constants.js';

export const COLLECTION_INDICATOR = `/*`;
const COLLECTION_ID = `collectionId`;

const FILERS_KEY = `filers`;
const FILING_STATUS_PATH = `/filingStatus`;
const MARRIED_FILING_JOINTLY_OPTION = `marriedFilingJointly`;
const MARRIED_FILING_SEPARATELY_OPTION = `marriedFilingSeparately`;
const FILING_STATUS_OPTIONS_TO_INCLUDE_SECONDARY_FILER = [
  MARRIED_FILING_JOINTLY_OPTION,
  MARRIED_FILING_SEPARATELY_OPTION,
];

export type ExportableFact = {
  value: string | null;
  sensitive: boolean;
};

export type ExportableCollectionItem = {
  [factKey: string]: ExportableFact;
};

type IndexedExportableCollectionItem = {
  [COLLECTION_ID]: string;
  facts: ExportableCollectionItem;
};

export type ExportableCollection = ExportableCollectionItem[];

export type ExportableFacts = {
  [key: string]: ExportableFact | ExportableCollection;
};

export const getStateExportableFactsFromGraph = (factGraph: FactGraph): ExportableFacts => {
  // TODO: Instead of this, next filing season, just call the backend API endpoint to get the exported values.
  //       (Requires EAG UWR).

  const exportableFacts = new Map<string, ExportableFact | IndexedExportableCollectionItem[]>();

  exportToStateFacts.forEach((exportableFactPath) => {
    const sensitive = isSensitive(exportableFactPath);

    if (Path.isAbstractPathType(exportableFactPath)) {
      // handle exportable facts with abstract paths (collections)
      const collectionIndicatorIndex = exportableFactPath.indexOf(COLLECTION_INDICATOR);
      const collectionBasePath = exportableFactPath.substring(0, collectionIndicatorIndex) as FDPath;
      if (!paths.includes(collectionBasePath)) {
        throw new Error(`Collection base path ${collectionBasePath} not present in fact dictionary.`);
      }
      const collectionKey = removeSlashes(collectionBasePath);

      const collectionResult = factGraph.get(Path.concretePath(collectionBasePath, null));
      const collectionUuids: string[] = collectionResult.hasValue
        ? scalaListToJsArray<string>(collectionResult.get.getItemsAsStrings())
        : [];

      collectionUuids.forEach((collectionUuid) => {
        const concretePath = Path.concretePath(exportableFactPath, collectionUuid);
        const factKey = getFactKeyFromPath(exportableFactPath) as Exclude<string, typeof COLLECTION_ID>;

        const factResult = factGraph.get(concretePath);
        const factValue = factResult.hasValue ? factResult.get.toString() : null;

        if (exportableFacts.has(collectionKey)) {
          // merge with existing collection
          const collection = exportableFacts.get(collectionKey);
          if (Array.isArray(collection)) {
            const indexToUpdate = collection.findIndex(
              (collectionItem) => collectionItem[COLLECTION_ID] === collectionUuid
            );
            const foundCollectionItemToUpdate = indexToUpdate >= 0;
            if (foundCollectionItemToUpdate) {
              const collectionItemToUpdate = collection[indexToUpdate];
              collectionItemToUpdate.facts = {
                ...collectionItemToUpdate.facts,
                [factKey]: {
                  value: factValue,
                  sensitive,
                },
              };
            } else {
              const collectionItem: IndexedExportableCollectionItem = {
                [COLLECTION_ID]: collectionUuid,
                facts: {
                  [factKey]: {
                    value: factValue,
                    sensitive,
                  },
                },
              };
              collection.push(collectionItem);
            }
          } else {
            // should never happen, but rather than just typecasting...
            throw new Error(`Collection ${collectionKey} not represented as an array`);
          }
        } else {
          // add a new collection, collection item, and first fact
          const collection: IndexedExportableCollectionItem[] = [];
          const collectionItem: IndexedExportableCollectionItem = {
            [COLLECTION_ID]: collectionUuid,
            facts: {
              [factKey]: {
                value: factValue,
                sensitive,
              },
            },
          };
          collection.push(collectionItem);

          exportableFacts.set(collectionKey, collection);
        }
      });
    } else {
      // handle exportable facts with absolute paths
      const concretePath = Path.concretePath(exportableFactPath, null);
      const factResult = factGraph.get(concretePath);
      const factKey = removeSlashes(concretePath);
      const factValue = factResult.hasValue ? factResult.get.toString() : null;
      const fact = {
        value: factValue,
        sensitive,
      };

      exportableFacts.set(factKey, fact);
    }
  });

  return sanitizeExportedOutput(exportableFacts, factGraph);
};

export const getFactKeyFromPath = (path: string) => {
  const abstractPathDelimiter = `/*/`;
  const lastIndexOfDelimiter = path.lastIndexOf(abstractPathDelimiter);

  if (lastIndexOfDelimiter > 0) {
    // get everything after the abstract path delimiter
    return removeSlashes(path.substring(lastIndexOfDelimiter + abstractPathDelimiter.length));
  }

  return removeSlashes(path);
};

const removeSlashes = (path: string) => {
  return (
    path
      // Replace any slash followed by a character with the uppercase character,
      .replaceAll(/\/[a-z]/g, (match, offset) => (offset > 0 ? match.substring(1).toUpperCase() : match))
      // then replace any remaining (leading or trailing) slash
      .replace(`/`, ``)
  );
};

const sanitizeExportedOutput = (
  exportableFactsMap: Map<string, ExportableFact | IndexedExportableCollectionItem[]>,
  factGraph: FactGraph
): ExportableFacts => {
  const sanitized: ExportableFacts = {};

  exportableFactsMap.forEach((value, key) => {
    if (Array.isArray(value)) {
      sanitized[key] = sanitizeCollection(key, value, factGraph);
    } else {
      sanitized[key] = sanitizeValue(value);
    }
  });

  return sanitized;
};

const EMPTY_STRING = ``;

const sanitizeCollection = (
  key: string,
  collection: IndexedExportableCollectionItem[],
  factGraph: FactGraph
): ExportableCollectionItem[] => {
  const sanitizedCollection = collection
    .map((collectionItem) => sanitizeCollectionItem(collectionItem))
    .filter((sanitizedCollectionItem) => !Object.values(sanitizedCollectionItem).every((value) => value === null));

  // Remove placeholder filer if not MFJ
  if (key === FILERS_KEY) {
    const filingStatus = factGraph.get(Path.concretePath(FILING_STATUS_PATH, null)).get.getValue();
    if (shouldFilterOutPlaceholderFiler(filingStatus)) {
      return sanitizedCollection.filter((ci) => ci.isPrimaryFiler.value === `true`);
    }
  }

  return sanitizedCollection;
};

const sanitizeCollectionItem = (indexedCollectionItem: IndexedExportableCollectionItem): ExportableCollectionItem => {
  const { [COLLECTION_ID]: _removedCollectionId, facts } = indexedCollectionItem;
  Object.entries(facts).forEach(([key, exportableFact]) => {
    if (exportableFact.value === EMPTY_STRING) {
      facts[key].value = null;
    }
  });

  return facts;
};

const sanitizeValue = (fact: ExportableFact): ExportableFact => {
  if (fact.value === EMPTY_STRING) {
    return { ...fact, value: null };
  } else {
    return fact;
  }
};

const shouldFilterOutPlaceholderFiler = (filingStatus: string) =>
  !FILING_STATUS_OPTIONS_TO_INCLUDE_SECONDARY_FILER.includes(filingStatus);

const isSensitive = (exportableFactPath: (typeof exportToStateFacts)[number]) =>
  SENSITIVE_EXPORTABLE_FACTS.includes(exportableFactPath);
