import { ScalaList, convertCollectionToArray } from '@irs/js-factgraph-scala';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { FlowCollectionLoop, FlowSubSubcategory, ItemAssertion } from '../../flow/flowConfig.js';
import { CardGroup } from '@trussworks/react-uswds';
import CollectionItemManager from '../../components/factTypes/CollectionItemManager/index.js';
import CollectionItem from '../../components/factTypes/CollectionItemList/CollectionItem.js';
import { Condition } from '../../flow/Condition.js';
import Subheading from '../../components/Subheading.js';
import {
  getEmptyAlertConfigs,
  getMefAlertConfigs,
  getTaxReturnAlertConfigs,
  MefAlertConfig,
} from '../../misc/aggregatedAlertHelpers.js';
import DFModal from '../../components/HelperText/DFModal.js';
import { createElement, useContext } from 'react';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import { conditionsPass } from '../../utils/condition.js';
import { SubSubCategory } from './SubSubCategory.js';
import { findFirstIncompleteScreenOfLoop } from '../../flow/flowHelpers.js';
import { AbsolutePath } from '../../fact-dictionary/Path.js';
import {
  InfoDisplayConfig,
  ScreenContentConfig,
  contentConfigIsInfoDisplayConfig,
} from '../../flow/ContentDeclarations.js';
import { InfoDisplayProps } from '../../types/core.js';
import { InfoTypeRenderer } from '../../components/factTypes/index.js';

interface CardGrouping {
  itemAssertions?: ItemAssertion[];
  hasIncompleteItems?: boolean;
  i18nKey?: string;
  i18nModalKey?: string;
  collectionItemIds: string[];
}

interface CollectionLoopRendererProps {
  loop: FlowCollectionLoop;
  subSubCategoryRefs: React.MutableRefObject<Map<string, React.MutableRefObject<HTMLHeadingElement>>>;
  allCollectionItems: string[];
  saveAndPersist: () => Promise<{ hasPersistError: boolean }>;
  sectionIsComplete: boolean;
  headingRefs: React.MutableRefObject<Map<string, React.MutableRefObject<HTMLHeadingElement>>>;
  collectionItemsWithAlerts: { errors: string[]; warnings: string[] };
  postContent: ScreenContentConfig[];
}

/**
 * This component renders the cards or the auto-iterating collection loop
 */
export function CollectionLoopRenderer({
  loop,
  saveAndPersist,
  subSubCategoryRefs,
  allCollectionItems,
  sectionIsComplete,
  headingRefs,
  collectionItemsWithAlerts,
  postContent,
}: CollectionLoopRendererProps) {
  const { factGraph } = useFactGraph();
  const result = factGraph.get(loop.collectionName);
  const { submissionStatus } = useContext(SubmissionStatusContext);

  if (loop.autoIterate) {
    // Render all content from loop on single page.
    const collectionItemIds = convertCollectionToArray(result.get as ScalaList<string>);
    return collectionItemIds.map((itemId: string) => {
      const nextIncompleteScreen = findFirstIncompleteScreenOfLoop(loop, factGraph, itemId);
      const indexOfNextIncompleteScreen = loop.subSubcategories.findIndex((ssc) => {
        return nextIncompleteScreen ? ssc.fullRoute === nextIncompleteScreen?.subSubcategoryRoute : false;
      });
      return (
        <>
          {loop.subSubcategories.map((ssc: FlowSubSubcategory, index) => {
            return (
              <SubSubCategory
                key={`${ssc.fullRoute}-${itemId}`}
                ssc={ssc}
                collectionId={itemId}
                refs={subSubCategoryRefs}
                headingContext={{ number: allCollectionItems.indexOf(itemId) + 1 }}
                isAfterNextIncompleteScreen={
                  indexOfNextIncompleteScreen === -1 ? false : index > indexOfNextIncompleteScreen
                }
                sectionIsComplete={sectionIsComplete}
                includesNextIncompleteScreen={index === indexOfNextIncompleteScreen}
                headingLevel={ssc.headingLevel}
                borderStyle={ssc.borderStyle}
              />
            );
          })}
        </>
      );
    });
  }

  const values: string[] = result.complete ? convertCollectionToArray(result.get as ScalaList<string>) : [];
  const cardGroupings: CardGrouping[] = loop.dataViewSections
    ? loop.dataViewSections.map((dvs) => {
        return {
          ...dvs,
          collectionItemIds: values.filter((itemId) => conditionsPass(dvs, factGraph, itemId)),
        };
      })
    : [{ collectionItemIds: values }];
  const cardGroupingHtml = cardGroupings
    // Do not render the group if it has no collectionitems.
    .filter((cg) => (cg.collectionItemIds.length === 0 ? null : cg))
    .map((cg) => (
      <div className='margin-top-5' key={`${loop.loopName}-${cg.i18nKey}`}>
        {cg.i18nKey && <Subheading i18nKey={cg.i18nKey} collectionId={null} />}
        {cg.i18nModalKey && (
          <div className='margin-bottom-2'>
            <DFModal i18nKey={cg.i18nModalKey} collectionId={null} />
          </div>
        )}
        <CardGroup>
          {cg.collectionItemIds.map((itemId, index) => {
            const isKnownToBeInProgress = loop.collectionItemCompletedCondition
              ? !new Condition(loop.collectionItemCompletedCondition).evaluate(factGraph, itemId)
              : false;
            const hideCardLabel2 = loop.hideCardLabel2Condition
              ? new Condition(loop.hideCardLabel2Condition).evaluate(factGraph, itemId)
              : false;
            const whereTo = `/data-view/loop/${encodeURIComponent(loop.loopName)}/${itemId}/?reviewMode=true`;
            const alertConfigs = getTaxReturnAlertConfigs(loop.screens, itemId, factGraph);
            const mefAlertConfigs = submissionStatus
              ? getMefAlertConfigs(loop.screens, itemId, factGraph, submissionStatus)
              : getEmptyAlertConfigs<MefAlertConfig>();
            const cardHeadingLevel = cg.i18nKey ? 3 : 2;
            const rawAssertion = cg.itemAssertions?.find((a) => conditionsPass(a, factGraph, itemId));
            const assertion = rawAssertion ? { i18nKey: rawAssertion.i18nKey, type: rawAssertion.type } : undefined;
            const aggregateErrorRoute =
              cg.i18nKey && itemId && collectionItemsWithAlerts.errors.includes(itemId) ? cg.i18nKey : undefined;
            const aggregateWarningRoute =
              cg.i18nKey && itemId && collectionItemsWithAlerts.warnings.includes(itemId) ? cg.i18nKey : undefined;
            const aggregateAlertKey = aggregateErrorRoute || aggregateWarningRoute;
            return (
              <CollectionItem
                itemPosition={index + 1}
                iconName={loop.iconName}
                loop={loop}
                detailLink={whereTo}
                itemId={itemId}
                path={loop.loopName}
                hideCardLabel2={hideCardLabel2}
                inProgress={isKnownToBeInProgress}
                alertConfigs={alertConfigs}
                mefAlertConfigs={mefAlertConfigs}
                key={itemId}
                cardHeadingLevel={cardHeadingLevel}
                assertion={assertion}
                refs={headingRefs}
                aggregateAlertKey={aggregateAlertKey}
              />
            );
          })}
        </CardGroup>
      </div>
    ));

  const postCardsContent = postContent
    .filter(
      (content): content is InfoDisplayConfig =>
        conditionsPass(content.props, factGraph, null) && contentConfigIsInfoDisplayConfig(content)
    )
    .map((content) => {
      const renderer = InfoTypeRenderer[content.componentName] as React.FunctionComponent<InfoDisplayProps>;
      return createElement(renderer, {
        ...(content.props as InfoDisplayProps),
        key: `${content.props.i18nKey}-${content.componentName}`,
      });
    });

  return (
    <>
      {cardGroupingHtml}
      {postContent.length > 0 && <div className='usa-prose'>{postCardsContent}</div>}
      <CollectionItemManager
        loopName={loop.loopName}
        loop={loop}
        donePath={loop.donePath}
        path={loop.collectionName as AbsolutePath}
        concretePath={loop.collectionName}
        saveAndPersist={saveAndPersist}
        knockoutRoute={loop.knockoutRoute}
        shouldSeeHubCompletionBtnsPath={loop.shouldSeeHubCompletionBtnsPath}
      />
    </>
  );
}
