import {
  ConcretePath,
  FactGraph,
  ScalaList,
  convertCollectionToArray,
  scalaListToJsArray,
} from '@irs/js-factgraph-scala';
import { ScreenConfig } from '../flow/ScreenConfig.js';
import { TaxReturnAlertDeclaration, MefAlertDeclaration, ScreenContentConfig } from '../flow/ContentDeclarations.js';
import { conditionsPass } from '../utils/condition.js';
import { TaxReturnSubmissionStatus } from '../types/core.js';
import { TaxReturnAlertProps } from '../components/Alert/TaxReturnAlert.js';
import { DFAlertProps } from '../components/Alert/DFAlert.js';
// TODO: Should this be using i18n from the useTranslate hook instead?
// eslint-disable-next-line no-restricted-imports
import i18n from 'i18next';
import { FlowCollectionLoop, FlowSubcategory } from '../flow/flowConfig.js';
import { useChecklistState } from '../flow/useChecklistState.js';
import { AggregateSummaryAlertProps } from '../components/SummaryAlert/AggregateSummaryAlert.js';
import {
  ScreenConfigWithCollectionId,
  findFirstIncompleteScreenOfLoop,
  hasAtLeastOneIncompleteCollectionItem,
} from '../flow/flowHelpers.js';
import { Condition } from '../flow/Condition.js';
import { assertNever } from 'assert-never';
import { getSscForLoop, getSectionRouteForScreen } from './collectionLoopHelpers.js';

export type SystemErrorConfig = {
  i18nKey: DFAlertProps[`i18nKey`];
};

export const getEmptyAlertConfigs = <
  AlertConfigType extends AlertConfig = AlertConfig
>(): AlertConfigs<AlertConfigType> => ({
  warnings: [],
  errors: [],
});

export interface AlertConfig {
  type: TaxReturnAlertProps['type'];
  route: string | undefined;
  subcategoryRoute?: string;
  subSubcategoryRoute?: string;
  checklistSubcategoryWarningLabel?: string;
  i18nKey: TaxReturnAlertProps['i18nKey'];
  isActive: boolean;
  loopName?: string | undefined;
  collectionId: TaxReturnAlertProps['collectionId'];
  factPaths?: string[];
}

export interface MefAlertConfig extends AlertConfig {
  mefErrorCode: string;
}

export type AlertConfigs<AlertConfigType extends AlertConfig = AlertConfig> = {
  warnings: AlertConfigType[];
  errors: AlertConfigType[];
};

export const filterAlertsBySubSubCategory = <AlertType extends AlertConfig>(
  alertConfigs: AlertConfigs<AlertType>,
  subSubCategoryFullRoute: string
) => ({
  warnings: alertConfigs.warnings.filter((wa) => wa && subSubCategoryFullRoute === wa.subSubcategoryRoute),
  errors: alertConfigs.errors.filter((ea) => ea && subSubCategoryFullRoute === ea.subSubcategoryRoute),
});

export const buildAlertKey = (alert: AlertConfig) => `${alert.i18nKey}-${alert.type}`;

export const buildDataViewAlertI18nKey = (alert: AlertConfig) =>
  i18n.exists(`${alert.i18nKey}/data-view`) ? `${alert.i18nKey}/data-view` : `dataviews.alerts.${alert.type}`;

export const getTaxReturnLoopAlerts = (categoryActive: boolean, loops: FlowCollectionLoop[], fg: FactGraph) => {
  const loopAlerts = getEmptyAlertConfigs();
  const alertConfigs = Object.assign(
    getEmptyAlertConfigs(),
    // Iterate through each loop
    ...loops.flatMap((loop: FlowCollectionLoop) => {
      const collectionItemsResult = fg.get(loop.collectionName);
      const collectionItems: string[] = collectionItemsResult.complete
        ? scalaListToJsArray(collectionItemsResult.get.getItemsAsStrings())
        : [];
      // For each collection item, return tax return alerts
      return collectionItems.map((itemId) => getTaxReturnAlertConfigs(loop.screens, itemId, fg, loopAlerts));
    })
  ) as AlertConfigs;

  return categoryActive ? alertConfigs : getEmptyAlertConfigs();
};

const getAggregatedAlertType = (alertType: string) => {
  switch (alertType) {
    case `warning`:
      return `warnings`;
    case `error`:
      return `errors`;
    case `info`:
      return `infos`;
    case `success`:
      return `successes`;
  }
};

export const getTaxReturnAlertConfigs = (
  screens: ScreenConfig[],
  collectionId: string | null,
  factGraph: FactGraph,
  existingAlerts?: AlertConfigs<AlertConfig>
): AlertConfigs => {
  const alertConfigs = existingAlerts || getEmptyAlertConfigs();

  screens.flatMap((s) => {
    const collection =
      s.collectionLoop && s.collectionContext ? factGraph.get(s.collectionContext as ConcretePath) : null;
    const collectionItemIds = collection?.complete ? convertCollectionToArray(collection.get as ScalaList<string>) : [];
    return s.content
      .filter((c) => {
        return c.componentName === `TaxReturnAlert`;
      })
      .forEach((c) => {
        const flowDeclaration = c.props as TaxReturnAlertDeclaration;
        const collectionItemIdPasses = collectionItemIds.some((id: string | null) =>
          conditionsPass(c.props, factGraph, id)
        );
        // If we have a collection id only use the props from the screen to determine if the alert is active.
        // Otherwise, we use this result of this OR check if any of the collection ids in the loop have active alerts.
        const flowDeclarationPasses = conditionsPass(flowDeclaration, factGraph, collectionId);
        const isActive = collectionId ? flowDeclarationPasses : flowDeclarationPasses || collectionItemIdPasses;
        if (isActive) {
          const alert: AlertConfig = {
            type: flowDeclaration.type,
            route: s.fullRoute(collectionId),
            subcategoryRoute: s.subcategoryRoute,
            subSubcategoryRoute: s.subSubcategoryRoute,
            i18nKey: flowDeclaration.i18nKey,
            isActive,
            loopName: s.collectionLoop?.loopName,
            collectionId: collectionItemIdPasses
              ? collectionItemIds.find((id: string | null) => conditionsPass(c.props, factGraph, id))
              : collectionId,
            factPaths: flowDeclaration.factPaths,
          };
          const alertIsInExistingAlerts =
            existingAlerts && alert
              ? existingAlerts[getAggregatedAlertType(alert.type) as keyof AlertConfigs].find(
                  (a: AlertConfig) => a.collectionId === alert.collectionId
                )
              : false;
          if (!alertIsInExistingAlerts) {
            switch (alert.type) {
              case `warning`:
                alertConfigs.warnings.push(alert);
                break;
              case `error`:
                alertConfigs.errors.push(alert);
                break;
              default:
                assertNever(alert.type);
            }
          }
        }
      });
  });

  return alertConfigs;
};

export const isRelevantMefAlert = (mefErrorCodes: string[], screenContentConfig: ScreenContentConfig): boolean => {
  const mefProps = screenContentConfig.props as MefAlertDeclaration;
  return screenContentConfig.componentName === `MefAlert` && mefErrorCodes.includes(mefProps.mefErrorCode);
};

export const getMefAlertConfigs = (
  screens: ScreenConfig[],
  collectionId: string | null,
  factGraph: FactGraph,
  submissionStatus: TaxReturnSubmissionStatus
): AlertConfigs<MefAlertConfig> => {
  const mefErrorCodes = submissionStatus.rejectionCodes.map((rc) => rc.MeFErrorCode);

  const mefAlertConfigs = getEmptyAlertConfigs<MefAlertConfig>();

  screens.flatMap((s) =>
    s.content
      .filter((c) => isRelevantMefAlert(mefErrorCodes, c))
      .forEach((c) => {
        const flowDeclaration = c.props as MefAlertDeclaration;
        const isActive = conditionsPass(flowDeclaration, factGraph, collectionId);
        if (isActive) {
          const mefAlertConfig: MefAlertConfig = {
            type: flowDeclaration.type,
            mefErrorCode: flowDeclaration.mefErrorCode,
            route: s.fullRoute(collectionId),
            subcategoryRoute: s.subcategoryRoute,
            subSubcategoryRoute: s.subSubcategoryRoute,
            i18nKey: flowDeclaration.i18nKey,
            isActive,
            collectionId: collectionId,
            factPaths: flowDeclaration.factPaths,
            loopName: undefined,
          };
          switch (mefAlertConfig.type) {
            case `warning`:
              mefAlertConfigs.warnings.push(mefAlertConfig);
              break;
            case `error`:
              mefAlertConfigs.errors.push(mefAlertConfig);
              break;
            default:
              assertNever(mefAlertConfig.type);
          }
        }
      })
  );

  return mefAlertConfigs;
};

const filterBySubCategoryRoute = <AlertConfigType extends AlertConfig>(
  alertConfigs: AlertConfigType[],
  subCategoryRoute: string
) => alertConfigs.filter((a) => a.subcategoryRoute === subCategoryRoute);

export const getSubCategoryAlertConfigs = <AlertConfigType extends AlertConfig>(
  alertConfigs: AlertConfigs<AlertConfigType>,
  subCategoryRoute: string
): AlertConfigs<AlertConfigType> => ({
  warnings: filterBySubCategoryRoute<AlertConfigType>(alertConfigs.warnings, subCategoryRoute),
  errors: filterBySubCategoryRoute<AlertConfigType>(alertConfigs.errors, subCategoryRoute),
});

const filterBySubSubCategory = <Alert extends AlertConfig>(alertConfigs: Alert[], subSubCategoryRoute: string) =>
  alertConfigs.filter((a) => a.subSubcategoryRoute === subSubCategoryRoute);

export const getSubSubCategoryAlertConfigs = <AlertConfigType extends AlertConfig>(
  alertConfigs: AlertConfigs<AlertConfigType>,
  subSubCategoryRoute: string
): AlertConfigs<AlertConfigType> => ({
  warnings: filterBySubSubCategory<AlertConfigType>(alertConfigs.warnings, subSubCategoryRoute),
  errors: filterBySubSubCategory<AlertConfigType>(alertConfigs.errors, subSubCategoryRoute),
});

export const extractUniqueAlertSummarySections = (checklistState: ReturnType<typeof useChecklistState>) => {
  const uniqueErrorSubCategoryRoutes = new Set<{ errorSubCategoryRoute: string; isIncomplete: boolean }>();
  const uniqueWarningSubCategoryRoutes = new Set<string>();
  checklistState
    .flatMap((c) => c.subcategories)
    .forEach((sc) => {
      const scErrors = [...sc.mefAlertConfigs.errors, ...sc.alertConfigs.errors];
      scErrors.forEach(
        (scError) =>
          scError.subcategoryRoute &&
          uniqueErrorSubCategoryRoutes.add({ errorSubCategoryRoute: scError.subcategoryRoute, isIncomplete: false })
      );

      const scWarnings = [...sc.mefAlertConfigs.warnings, ...sc.alertConfigs.warnings];
      scWarnings.forEach(
        (scWarning) => scWarning.subcategoryRoute && uniqueWarningSubCategoryRoutes.add(scWarning.subcategoryRoute)
      );

      if (sc.hasIncompletedCollectionItem || sc.isStartedButNotComplete) {
        uniqueErrorSubCategoryRoutes.add({ errorSubCategoryRoute: sc.subcategoryRoute, isIncomplete: true });
      }
    });
  const summaryErrorSections: AggregateSummaryAlertProps['summaryErrorSections'] = [];
  uniqueErrorSubCategoryRoutes.forEach((error) => {
    summaryErrorSections.push({
      i18nKey: `checklist.${error.errorSubCategoryRoute}.heading`,
      path: `${error.errorSubCategoryRoute}`,
      isIncomplete: error.isIncomplete,
    });
  });
  const summaryWarningSections: AggregateSummaryAlertProps['summaryWarningSections'] = [];
  uniqueWarningSubCategoryRoutes.forEach((warningSubCategoryRoute) => {
    summaryWarningSections.push({
      i18nKey: `checklist.${warningSubCategoryRoute}.heading`,
      path: `${warningSubCategoryRoute}`,
    });
  });

  return {
    summaryErrorSections,
    summaryWarningSections,
  };
};

export const getSubSubCategoryI18nKey = (
  subcategoryRoute: string | undefined,
  subSubCategoryRouteSuffix: string | undefined,
  i18n: { exists: (key: string) => boolean }
) => {
  const dataViewErrorKey = `dataviews.subsubcategories.${subcategoryRoute}.${subSubCategoryRouteSuffix}`;
  const dataViewErrorKeyExists = i18n.exists(dataViewErrorKey);
  return dataViewErrorKeyExists
    ? dataViewErrorKey
    : `subsubcategories.${subcategoryRoute}.${subSubCategoryRouteSuffix}`;
};

/**  Map each of the alerts in the summary to the heading they should link to
 */
export const extractUniqueAlertSummarySectionsForDataView = (
  mefAlertsConfigs: AlertConfigs<MefAlertConfig>,
  taxReturnAlertConfigs: AlertConfigs<AlertConfig>,
  subcategoryOrLoop: FlowSubcategory | FlowCollectionLoop,
  incompleteScreen: ScreenConfigWithCollectionId | undefined
): {
  summaryErrorSections: { i18nKey: string; path: string; isIncomplete: boolean }[];
  summaryWarningSections: { i18nKey: string; path: string }[];
  collectionItemsWithAlerts: { errors: never[]; warnings: never[] };
} => {
  // Aggregate all the warnings and errors
  const warningConfigs = [...mefAlertsConfigs.warnings, ...taxReturnAlertConfigs.warnings];
  const errorConfigs = [...mefAlertsConfigs.errors, ...taxReturnAlertConfigs.errors];

  // Map each warning to the link associated.
  const summaryWarningSections = warningConfigs.map((w: AlertConfig) => {
    const ssc = w.loopName
      ? getSscForLoop(w, subcategoryOrLoop)
      : subcategoryOrLoop.subSubcategories.find((ssc) => ssc.fullRoute === w.subSubcategoryRoute);

    return {
      i18nKey: getSubSubCategoryI18nKey(w.subcategoryRoute, ssc?.routeSuffix, i18n),
      path: `${ssc?.routeSuffix}`,
    };
  });

  // Map each error to the link associated.
  const summaryErrorSections = errorConfigs.map((e: AlertConfig) => {
    const ssc = e.loopName
      ? getSscForLoop(e, subcategoryOrLoop)
      : subcategoryOrLoop.subSubcategories.find((ssc) => ssc.fullRoute === e.subSubcategoryRoute);
    return {
      i18nKey: getSubSubCategoryI18nKey(e.subcategoryRoute, ssc?.routeSuffix, i18n),
      path: `${ssc?.routeSuffix}`,
      isIncomplete: false,
    };
  });

  if (incompleteScreen?.screen) {
    const { screen } = incompleteScreen;
    const route = getSectionRouteForScreen(screen, subcategoryOrLoop);
    const routeArray = route.split(`/`);
    const routeSuffix = routeArray[routeArray.length - 1];
    summaryErrorSections.push({
      i18nKey: getSubSubCategoryI18nKey(screen.subcategoryRoute, routeSuffix, i18n),
      path: routeSuffix,
      isIncomplete: true,
    });
  }

  return {
    summaryErrorSections,
    summaryWarningSections,
    collectionItemsWithAlerts: {
      errors: [],
      warnings: [],
    },
  };
};

const getMatchingDataViewSection = (factGraph: FactGraph, loops: FlowCollectionLoop[], alertConfig: AlertConfig) => {
  const loop = loops.find((loop) => loop.loopName === alertConfig.loopName);
  return loop?.dataViewSections?.find((dvs) => conditionsPass(dvs, factGraph, alertConfig.collectionId));
};

const getIncompleteScreensOfLoop = (loop: FlowCollectionLoop, factGraph: FactGraph) => {
  const incompleteScreens: { screen: ScreenConfig; id: string | null; i18nKey: string | undefined }[] = [];
  if (hasAtLeastOneIncompleteCollectionItem(loop, factGraph)) {
    const result = factGraph.get(loop.collectionName);
    const collectionItemIds = convertCollectionToArray(result.get as ScalaList<string>);
    for (const itemId of collectionItemIds) {
      const foundScreen = findFirstIncompleteScreenOfLoop(loop, factGraph, itemId);
      const matchingDataViewSection = loop.dataViewSections?.find((dvs) => conditionsPass(dvs, factGraph, itemId));
      if (foundScreen) {
        incompleteScreens.push({
          screen: foundScreen,
          id: itemId,
          i18nKey: matchingDataViewSection?.i18nKey,
        });
      }
    }
  }
  return incompleteScreens;
};

export const extractUniqueAlertSummarySectionsForCollectionHub = (
  factGraph: FactGraph,
  mefAlertsConfigs: AlertConfigs<MefAlertConfig>,
  taxReturnAlertConfigs: AlertConfigs<AlertConfig>,
  collectionHubLoop: FlowCollectionLoop
) => {
  const warningConfigs = [...mefAlertsConfigs.warnings, ...taxReturnAlertConfigs.warnings];
  const errorConfigs = [...mefAlertsConfigs.errors, ...taxReturnAlertConfigs.errors];

  const uniqueErrorSectionRoutes = new Set<{ itemId: string | null; key: string; isIncomplete: boolean }>();
  const uniqueWarningSectionRoutes = new Set<{ itemId: string | null; key: string }>();

  // Map each warning to a matching dataview section
  warningConfigs.forEach((w) => {
    const matchingDataViewSection = getMatchingDataViewSection(factGraph, [collectionHubLoop], w);
    return (
      matchingDataViewSection &&
      uniqueWarningSectionRoutes.add({ itemId: w.collectionId, key: matchingDataViewSection.i18nKey })
    );
  });

  // Map each error to a matching dataview section
  errorConfigs.forEach((e) => {
    const matchingDataViewSection = getMatchingDataViewSection(factGraph, [collectionHubLoop], e);
    return (
      matchingDataViewSection &&
      uniqueErrorSectionRoutes.add({
        itemId: e.collectionId,
        key: matchingDataViewSection.i18nKey,
        isIncomplete: false,
      })
    );
  });

  // Map one incomplete screen per card to associated section
  const incompleteScreens = getIncompleteScreensOfLoop(collectionHubLoop, factGraph);
  if (incompleteScreens.length) {
    incompleteScreens.forEach((s) => {
      const { id } = s;
      const collectionItemComplete = collectionHubLoop?.collectionItemCompletedCondition
        ? new Condition(collectionHubLoop.collectionItemCompletedCondition).evaluate(factGraph, id)
        : false;
      // Only get matching data view section if the collection item is incomplete.
      const matchingDataViewSection = !collectionItemComplete
        ? collectionHubLoop?.dataViewSections?.find((dvs) => conditionsPass(dvs, factGraph, id))
        : undefined;
      matchingDataViewSection &&
        uniqueErrorSectionRoutes.add({ itemId: id, key: matchingDataViewSection?.i18nKey, isIncomplete: true });
    });
  }

  const summaryErrorSections: AggregateSummaryAlertProps['summaryErrorSections'] = [];
  uniqueErrorSectionRoutes.forEach((s) => {
    summaryErrorSections.push({
      i18nKey: s.key,
      path: s.key,
      isIncomplete: s.isIncomplete,
    });
  });
  const summaryWarningSections: AggregateSummaryAlertProps['summaryWarningSections'] = [];
  uniqueWarningSectionRoutes.forEach((s) => {
    summaryWarningSections.push({
      i18nKey: s.key,
      path: s.key,
    });
  });

  const collectionItemsWithWarnings = Array.from(uniqueWarningSectionRoutes).map((s) => (s.itemId ? s.itemId : ``));
  const collectionItemsWithErrors = Array.from(uniqueErrorSectionRoutes).map((s) => (s.itemId ? s.itemId : ``));
  return {
    summaryErrorSections,
    summaryWarningSections,
    collectionItemsWithAlerts: {
      errors: collectionItemsWithErrors,
      warnings: collectionItemsWithWarnings,
    },
  };
};
