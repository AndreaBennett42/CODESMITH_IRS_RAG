import { TaxReturn, TaxReturnSubmissionStatus } from '../types/core.js';
import { FlowCollectionLoop, FlowConfig, useFlow } from './flowConfig.js';
import { FactGraph } from '@irs/js-factgraph-scala';
import { Condition } from './Condition.js';
import { useFactGraph } from '../factgraph/FactGraphContext.js';
import {
  AlertConfigs,
  getTaxReturnAlertConfigs,
  getMefAlertConfigs,
  getEmptyAlertConfigs,
  MefAlertConfig,
  getTaxReturnLoopAlerts,
} from '../misc/aggregatedAlertHelpers.js';
import { hasAtLeastOneIncompleteCollectionItem, subcategoryHasSomeCompletedFacts } from './flowHelpers.js';
import { getUrlSearchParams } from '../screens/navUtils.js';
import { getCollectionId } from '../screens/BaseScreen.js';
import { ChecklistSubcategoryProps } from '../components/checklist/ChecklistSubcategory/ChecklistSubcategory.js';
import { useContext } from 'react';
import { SubmissionStatusContext } from '../context/SubmissionStatusContext/SubmissionStatusContext.js';

export interface ChecklistProps {
  taxReturns: TaxReturn[];
}

const KNOCKOUT_CATEGORY_ROUTE = `/flow/knockout`;
export const DEFAULT_EXCLUDED_CATEGORIES = new Set([KNOCKOUT_CATEGORY_ROUTE]);

export function useChecklistState(excludedCategories = DEFAULT_EXCLUDED_CATEGORIES) {
  const { factGraph } = useFactGraph();

  factGraph.save();
  const flow = useFlow();
  const { submissionStatus } = useContext(SubmissionStatusContext);
  return getChecklistState(factGraph, flow, submissionStatus, excludedCategories);
}

export function getChecklistState(
  factGraph: FactGraph,
  flow: FlowConfig,
  submissionStatus?: TaxReturnSubmissionStatus,
  excludedCategories = DEFAULT_EXCLUDED_CATEGORIES
) {
  let prevSubcategoryComplete = true;
  return (
    flow.categories
      // We don't render the knockout category
      .filter((c) => !excludedCategories.has(c.route))
      .map((cat) => {
        let categoryActive = false;
        return {
          route: cat.route,
          subcategories: cat.subcategories
            .map((sc) => {
              // We don't need our actual search params (we're on the checklist, which shouldn't have a collection id),
              // but we do need to get a collection Id for either primary filer or spouse.
              const collectionId = getCollectionId(factGraph, new URLSearchParams(), sc.collectionName);

              const shouldHideSectionFunction = () => {
                if (sc.displayOnlyIf) {
                  if (Array.isArray(sc.displayOnlyIf)) {
                    return sc.displayOnlyIf.some(
                      (condtion) => new Condition(condtion).evaluate(factGraph, collectionId) === false
                    );
                  } else {
                    return new Condition(sc.displayOnlyIf).evaluate(factGraph, collectionId) === false;
                  }
                } else {
                  return false;
                }
              };

              if (shouldHideSectionFunction()) {
                return null;
              }

              // We may have multiple conditions, so we evaluate all of them to see if the category is complete.
              const allCompletionConditions = Array.isArray(sc.completionCondition)
                ? sc.completionCondition
                : [sc.completionCondition];

              // Find available loops
              const availableLoops = sc.loops.filter((loop: FlowCollectionLoop) => {
                const availableScreens = loop.screens.filter((c) => c.isAvailable(factGraph, collectionId));
                // eslint-disable-next-line eqeqeq
                return availableScreens.length != 0;
              });

              // Check if any loops have incomplete collection items
              const hasIncompletedCollectionItem =
                categoryActive &&
                prevSubcategoryComplete &&
                // Section has been started
                subcategoryHasSomeCompletedFacts(sc, factGraph, collectionId) &&
                availableLoops.some((loop: FlowCollectionLoop) =>
                  hasAtLeastOneIncompleteCollectionItem(loop, factGraph)
                );

              const markIncompleteDueToIncompleteCollectionItem =
                sc.lockFutureSectionsIfCollectionItemsIncomplete && hasIncompletedCollectionItem;

              // We only calculate completeness from our conditions if the previous category is already complete.
              // If the previous category is not complete, this subcategory won't be complete yet either.
              const isComplete = prevSubcategoryComplete
                ? allCompletionConditions.every((condition) =>
                    new Condition(condition).evaluate(factGraph, collectionId)
                  ) && !markIncompleteDueToIncompleteCollectionItem
                : false;
              let isNext = prevSubcategoryComplete && !isComplete;
              if (isNext) {
                prevSubcategoryComplete = false;
              }
              if (isNext || isComplete) {
                categoryActive = true;
              }

              const isStartedButNotComplete =
                isNext && !isComplete && subcategoryHasSomeCompletedFacts(sc, factGraph, collectionId);
              const getNavigationUrl = () => {
                if (!categoryActive) {
                  return undefined;
                } else if (sc.hasDataView && (isComplete || isStartedButNotComplete)) {
                  const dataViewOverrideScreen = sc.screens
                    .filter((s) => s.actAsDataView)
                    .find((s) => s.isAvailable(factGraph, collectionId));
                  const urlSearchParams = getUrlSearchParams(null, sc.collectionName, collectionId, true);
                  if (dataViewOverrideScreen)
                    return `${dataViewOverrideScreen.fullRoute(collectionId)}${
                      urlSearchParams && `?`
                    }${urlSearchParams}`;

                  return `/data-view/${sc.route}${urlSearchParams && `?`}${urlSearchParams}`;
                } else {
                  // The below code finds the first available route.
                  return sc.screens.find((sc) => sc.isAvailable(factGraph, collectionId))?.fullRoute(collectionId);
                }
              };
              const navigationUrl = getNavigationUrl();

              // eslint-disable-next-line eqeqeq
              if (isNext && navigationUrl == undefined) {
                // this subcategory will be hidden, so it will not actually be next!
                // We need to mark this subcategory back as complete and not next.
                isNext = false;
                prevSubcategoryComplete = true;
              }

              const subcategoryAlerts = categoryActive
                ? getTaxReturnAlertConfigs(sc.screens, collectionId, factGraph)
                : getEmptyAlertConfigs();

              const loopAlerts = getTaxReturnLoopAlerts(categoryActive, sc.loops, factGraph);
              const mergedAlerts: AlertConfigs = {
                warnings: [...subcategoryAlerts.warnings, ...loopAlerts.warnings],
                errors: [...subcategoryAlerts.errors, ...loopAlerts.errors],
              };

              const hasSubmissionRejectionErrors = submissionStatus && submissionStatus.rejectionCodes.length > 0;
              const subCategoryMefAlerts = hasSubmissionRejectionErrors
                ? getMefAlertConfigs(sc.screens, collectionId, factGraph, submissionStatus)
                : getEmptyAlertConfigs<MefAlertConfig>();

              return {
                subcategoryRoute: sc.route,
                isNext,
                isStartedButNotComplete,
                isComplete: isComplete,
                hasIncompletedCollectionItem,
                navigationUrl: navigationUrl,
                alertConfigs: mergedAlerts,
                mefAlertConfigs: subCategoryMefAlerts,
                dataItems: sc.dataItems,
              } as ChecklistSubcategoryProps;
            })
            // Filter out any hidden sections
            // eslint-disable-next-line eqeqeq
            .filter((subcat): subcat is ChecklistSubcategoryProps => subcat != null),
          categoryActive,
        };
      })
  );
}
