/* eslint-disable df-rules/no-factgraph-save */
import { useAppSelector } from '../../redux/hooks.js';
import { Button } from '@trussworks/react-uswds';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { useFlow } from '../../flow/flowConfig.js';
import { contentConfigIsFactConfig } from '../../flow/ContentDeclarations.js';
import { saveImportedFacts } from '../../redux/slices/data-import/saveImportedFacts.js';
import { CollectionFactory, ConcretePath, scalaListToJsArray } from '@irs/js-factgraph-scala';
import { useSaveAndPersist } from '../../hooks/useSaveAndPersist.js';
import { setFactsFromFactActionPaths } from '../screenUtils.js';
import Translation from '../Translation/Translation.js';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Path } from '../../flow/Path.js';
import styles from './CollectionDataPreview.module.scss';
import { SetFactActionConfig } from '../../flow/ScreenConfig.js';

import {
  CollectionItemPreviewTile,
  CollectionItemProps,
  getCollectionItemContent,
} from './CollectionItemPreviewTile.js';
import {
  getDataImportCollectionKey,
  setImportedMetaDataFacts,
  setOfferedMetadataFacts,
} from './collectionDataPreviewHelpers.js';
import { getFirstAvailableOfCollectionLoop } from '../../misc/getCollectionLoopEnds.js';
import { useTranslation } from 'react-i18next';

export type CollectionDataPreviewProps = {
  collectionContext: ConcretePath;
  nextRouteOverride: string;
};

const CollectionDataPreview = ({ collectionContext, nextRouteOverride }: CollectionDataPreviewProps) => {
  const { factGraph } = useFactGraph();
  const flow = useFlow();
  const saveAndPersist = useSaveAndPersist();
  const [itemChecked, setItemChecked] = useState(new Map<string, boolean>());
  const navigate = useNavigate();
  const { i18n } = useTranslation();

  const updateItemChecked = (key: string, value: boolean) => {
    setItemChecked((prevMap) => new Map(prevMap).set(key, value));
  };

  const collectionLoop = flow.collectionLoopsByName.get(collectionContext);
  const dataImportSlice = useAppSelector((state) => state.dataImportProfile);
  const { data } = dataImportSlice;
  const isComplete = data.status === `complete`;
  const profile = isComplete ? data.profile.data : null;
  const profileKey = getDataImportCollectionKey(collectionContext) as keyof typeof profile;

  const dataImportCollection = profile ? profile[profileKey] : { state: `incomplete`, payload: [] };
  const collectionPayload =
    isComplete && dataImportCollection.state === `success` ? dataImportCollection.payload : null;
  const collectionArray = (collectionPayload as []) || [];

  setOfferedMetadataFacts(collectionContext, collectionArray, factGraph);

  if (!isComplete || !data.profile) {
    // TODO - actual error handling when designs come in.
    return;
  }
  if (!collectionLoop) {
    // TODO - actual error handling when designs come in.
    return;
  }

  const collectionResult = factGraph.get(collectionContext as ConcretePath);
  const collectionItems: string[] = collectionResult.complete
    ? scalaListToJsArray(collectionResult.get.getItemsAsStrings())
    : [];

  const setFacts = async () => {
    // Initialize the new collection.
    const newItemIds = Array.from(itemChecked)
      .filter((entry) => entry[1] === true && !collectionItems.includes(entry[0]))
      .map((entry) => entry[0]);
    const newCollectionItems = [...collectionItems, ...newItemIds];
    factGraph.set(collectionContext as ConcretePath, CollectionFactory(newCollectionItems));
    factGraph.save();

    newItemIds.forEach((item) => {
      // Set the isImported Fact for this item.
      if (collectionLoop.isImportedFactPath)
        factGraph.set(Path.concretePath(collectionLoop.isImportedFactPath, item), true);
      factGraph.save();
      collectionLoop?.screens?.forEach(async (s) => {
        const factsToImport = s?.content
          .filter(contentConfigIsFactConfig)
          .filter((child) => child.props.path && child.props.importedPath);
        if (factsToImport.length) {
          saveImportedFacts(data.profile, factGraph, factsToImport, item);
        }
        // On every available screen in the loop,
        // set the facts that get set upon visiting that screen.
        // except any fact actions that include the "done path" for the imported flow.
        if (s.isAvailable(factGraph, item) && s.setActions.length) {
          const filterOutImportedFlowDonePath = (action: SetFactActionConfig) =>
            collectionLoop.importedFlowDonePath ? !action.path.includes(collectionLoop.importedFlowDonePath) : null;
          const actions = s.setActions.filter(filterOutImportedFlowDonePath);
          setFactsFromFactActionPaths(factGraph, item, actions);
        }
      });
      setImportedMetaDataFacts(profileKey, factGraph, item);
    });
    factGraph.save();
  };

  const handleCheckboxClick = async (event: React.FormEvent<HTMLInputElement>) => {
    const checkbox = event.target as HTMLInputElement;
    updateItemChecked(checkbox.id, checkbox.checked);
  };

  const disableImportButton = [...itemChecked.values()].every((value) => value === false);
  const multipleItemsChecked = [...itemChecked.values()].filter((isChecked) => isChecked === true).length > 1;

  return (
    <div>
      <fieldset className='usa-fieldset padding-bottom-3' aria-labelledby='page-heading'>
        <ul className='usa-list usa-list--unstyled'>
          {collectionArray.map((collection: CollectionItemProps, i: number) => {
            const itemWasChecked = itemChecked.get(collection.id) || false;
            const { header: itemHeader } = getCollectionItemContent(i18n, collectionContext, collection);
            return (
              <li
                key={collection.id}
                className={`${styles.collectionDataPreviewItem} ${
                  itemWasChecked ? styles.collectionDataPreviewItemChecked : null
                }`}
              >
                <div data-testid='checkbox' className={`usa-checkbox`}>
                  <input
                    className={`usa-checkbox__input margin-top-4`}
                    id={collection.id}
                    onClick={handleCheckboxClick}
                    type='checkbox'
                    defaultChecked={itemWasChecked}
                    aria-describedby={`item-description-${i}`}
                  />
                  <label className='usa-checkbox__label' htmlFor={collection.id}>
                    <strong>{itemHeader}</strong>
                  </label>
                </div>
                <CollectionItemPreviewTile
                  id={`item-description-${i}`}
                  collectionContext={collectionContext}
                  collection={collection}
                />
              </li>
            );
          })}
        </ul>
      </fieldset>
      <div>
        <Button
          disabled={disableImportButton}
          type='submit'
          onClick={async (event: React.MouseEvent<HTMLButtonElement>) => {
            event.preventDefault();
            await setFacts();
            let hasPersistError = false;
            const persistResult = await saveAndPersist();
            hasPersistError = persistResult.hasPersistError;
            if (!hasPersistError) {
              const importedCollectionIds = Array.from(itemChecked)
                .filter((entry) => entry[1] === true)
                .map((entry) => entry[0]);
              const whereTo = getFirstAvailableOfCollectionLoop(
                collectionLoop.loopName,
                factGraph,
                importedCollectionIds[0],
                flow
              );
              if (!whereTo) throw new Error(`Cannot find a next path`);
              navigate(whereTo.routable.fullRoute(whereTo.collectionId, { reviewMode: false }));
            }
          }}
        >
          <Translation
            i18nKey={`datapreviews.${collectionContext}.button.${
              multipleItemsChecked ? `importPlural` : `importSingular`
            }`}
            collectionId={null}
          />
        </Button>
      </div>
      <div className='screen__actions'>
        <Button
          outline
          type='submit'
          onClick={async (event: React.MouseEvent<HTMLButtonElement>) => {
            event.preventDefault();
            factGraph.delete(collectionContext as ConcretePath);
            await saveAndPersist();
            navigate(nextRouteOverride);
          }}
        >
          <Translation i18nKey={`datapreviews.${collectionContext}.button.doNotImport`} collectionId={null} />
        </Button>
      </div>
    </div>
  );
};

export default CollectionDataPreview;
