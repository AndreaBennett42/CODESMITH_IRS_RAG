import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { FlowCollectionLoop, FlowConfig, FlowSubSubcategory } from '../../flow/flowConfig.js';
import { MutableRefObject, RefObject } from 'react';
import { useSearchParams, Link } from 'react-router-dom';

import { CollectionLoopRenderer } from './CollectionLoopRenderer.js';
import { hasAtLeastOneIncompleteCollectionItem } from '../../flow/flowHelpers.js';
import Translation from '../../components/Translation/index.js';

import { useSaveAndPersistIfPossible } from '../../hooks/useSaveAndPersistIfPossible.js';
import { ScalaList, convertCollectionToArray } from '@irs/js-factgraph-scala';
import Screen from '../../components/Screen.js';
import { ScreenContentConfig } from '../../flow/ContentDeclarations.js';
import { ScreenConfig } from '../../flow/ScreenConfig.js';
import { getCollectionId } from '../BaseScreen.js';
import { Icon } from '@trussworks/react-uswds';
import { handleRefFromRoute } from '../../components/SummaryAlert/summaryHelpers.js';
import styles from './CollectionHubDataView.module.scss';
import DataReveal from '../../components/checklist/ChecklistSubcategory/DataReveal.js';

type CollectionHubDataViewProps = {
  collectionHub?: ScreenConfig; // The screen with cards
  collectionLoop: FlowCollectionLoop; // The collection loop
  variant: 'full' | 'nested' | 'flat'; // Which variant/style
  // Used for autoiterating or nested refs
  subSubCategoryRefs: MutableRefObject<Map<string, MutableRefObject<HTMLHeadingElement>>>;
  // Used for headings of cards refs // why two?
  headingRefs: MutableRefObject<Map<string, MutableRefObject<HTMLHeadingElement>>>;
  flow: FlowConfig;
};

type CollectionHubNestedHeaderProps = {
  ssc: FlowSubSubcategory;
  refs: React.MutableRefObject<Map<string, React.MutableRefObject<HTMLHeadingElement>>>;
  collectionLoop: FlowCollectionLoop;
};

/**
 * This is for the header of the nested entries
 * Should match styling for SubSubCategoryHeaderProps (but not exact)
 */
export const CollectionHubNestedHeader = ({ ssc, collectionLoop, refs }: CollectionHubNestedHeaderProps) => {
  const headeri18nKey = `dataviews.${collectionLoop.fullRoute}.heading`;

  return (
    <div className={styles.nestedSectionHeader}>
      <Link
        ref={handleRefFromRoute(ssc.routeSuffix, refs) as RefObject<HTMLAnchorElement>}
        to={`/data-view${collectionLoop.fullRoute}`}
      >
        <h2 id={ssc.routeSuffix} className={styles.nestedSectionHeading}>
          <Translation i18nKey={headeri18nKey} collectionId={null} />
        </h2>
        <Icon.NavigateNext aria-hidden='true' className='margin-right-05' size={3} />
      </Link>
    </div>
  );
};

const CollectionHubDataView = ({
  collectionHub,
  collectionLoop,
  variant: displayLevel,
  subSubCategoryRefs,
  headingRefs,
  flow,
}: CollectionHubDataViewProps) => {
  const { factGraph } = useFactGraph();
  const [searchParams] = useSearchParams();

  const saveAndPersistIfPossible = useSaveAndPersistIfPossible();

  // Only inner loops can be nested
  // eslint-disable-next-line eqeqeq
  if (displayLevel == `nested` && !collectionLoop.isInner) {
    throw new Error(`Nested collection loop ${collectionLoop.loopName} must be inner loop`);
  }

  // Get the whole subcategory object from the flow
  const subsubcategory = collectionLoop.isInner
    ? flow.subsubcategoriesByRoute.get(collectionLoop.fullRoute)
    : undefined;

  // If no available screens, don't show
  if (subsubcategory && collectionLoop.isInner) {
    const availableScreens = subsubcategory.screens.filter((c) => c.isAvailable(factGraph, null));
    // eslint-disable-next-line eqeqeq
    if (!availableScreens || availableScreens.length == 0) {
      return null;
    }
  }

  const collectionId = collectionLoop.collectionName
    ? getCollectionId(factGraph, searchParams, collectionLoop.collectionName)
    : null;

  // Get the ids from all items in the collection?
  const allCollectionItems = () => {
    const result = factGraph.get(collectionLoop.collectionName);
    const collectionItemIds = result.complete ? convertCollectionToArray(result.get as ScalaList<string>) : [];
    return collectionItemIds;
  };
  const isComplete = !hasAtLeastOneIncompleteCollectionItem(collectionLoop, factGraph);

  // Full hub dataview - We want the same content as the collection hub content in the flow.
  // This means we render the content above the cards.
  let collectionHubContentBeforeCollection: ScreenContentConfig[] = [];
  let collectionHubContentAfterCollection: ScreenContentConfig[] = [];
  if (collectionHub) {
    // We don't want to render the cards in this component because CollectionLoopRenderer will handle this below.
    // This collects the content before and after the cards.
    const collectionManagerIndex = collectionHub.content.findIndex((s) => s.componentName === `CollectionItemManager`);
    if (collectionManagerIndex !== -1) {
      collectionHubContentBeforeCollection = collectionHub.content.slice(0, collectionManagerIndex);
      collectionHubContentAfterCollection = collectionHub.content.slice(collectionManagerIndex + 1);
    }
  }

  const collectionItemsWithAlerts = { errors: [], warnings: [] };
  return (
    <>
      {
        // eslint-disable-next-line eqeqeq
        displayLevel == `nested` && collectionLoop.isInner && subsubcategory ? (
          // collection loop nested view where you can click through
          <div className={`${styles.dataViewSection} padding-bottom-0 margin-top-0`}>
            <CollectionHubNestedHeader ssc={subsubcategory} collectionLoop={collectionLoop} refs={subSubCategoryRefs} />
            <DataReveal
              dataItems={collectionLoop.dataReveals}
              factGraph={factGraph}
              subcategoryRoute={collectionLoop.fullRoute}
              collectionId={collectionId}
              i18nKey={`dataviews.${collectionLoop.fullRoute}.dataReveals`}
            />
          </div>
        ) : (
          // collection loop autoiterate, where each item becomes a little section
          <>
            {
              // eslint-disable-next-line eqeqeq
              displayLevel == `full` && collectionHub && (
                // Make a fake screen with pre-collection-hub content?
                <Screen
                  screenRoute={collectionHub.route}
                  screenContent={collectionHubContentBeforeCollection}
                  collectionId={collectionId}
                  gotoNextScreen={() => null}
                  setFactActionPaths={[]}
                  key={collectionHub.fullRoute(collectionId)}
                />
              )
            }
            <CollectionLoopRenderer
              loop={collectionLoop}
              saveAndPersist={saveAndPersistIfPossible}
              allCollectionItems={allCollectionItems()}
              // one is used for autoiterating and one is used for collection hub view
              subSubCategoryRefs={subSubCategoryRefs}
              sectionIsComplete={isComplete}
              headingRefs={headingRefs}
              collectionItemsWithAlerts={collectionItemsWithAlerts}
              postContent={collectionHubContentAfterCollection}
            />
          </>
        )
      }
    </>
  );
};

export default CollectionHubDataView;
