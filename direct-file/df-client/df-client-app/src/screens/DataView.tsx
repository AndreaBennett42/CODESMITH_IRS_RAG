import { useFactGraph } from '../factgraph/FactGraphContext.js';
import { FlowCollectionLoop, FlowSubSubcategory, useFlow } from '../flow/flowConfig.js';
import { useMemo, useRef, useEffect, MutableRefObject, useContext } from 'react';
import { Link, useParams, useSearchParams, useLocation } from 'react-router-dom';
import { getCollectionId } from './BaseScreen.js';
import {
  AlertConfig,
  buildAlertKey,
  buildDataViewAlertI18nKey,
  extractUniqueAlertSummarySectionsForDataView,
  extractUniqueAlertSummarySectionsForCollectionHub,
  getEmptyAlertConfigs,
  getMefAlertConfigs,
  getSubSubCategoryAlertConfigs,
  getTaxReturnAlertConfigs,
  MefAlertConfig,
} from '../misc/aggregatedAlertHelpers.js';
import { SubSubCategory } from './data-view/SubSubCategory.js';
import { Alert } from '@trussworks/react-uswds';
import PageTitle from '../components/PageTitle/index.js';
import Translation from '../components/Translation/index.js';
import { Condition } from '../flow/Condition.js';
import { findFirstIncompleteScreenOfSubcategory, hasAtLeastOneIncompleteCollectionItem } from '../flow/flowHelpers.js';
import { useTranslation } from 'react-i18next';
import { conditionsPass } from '../utils/condition.js';
import { DataViewDynamicNav } from './data-view/DataViewDynamicNav.js';
import SectionsAlertAggregator from '../components/SectionsAlertAggregator/SectionsAlertAggregator.js';
import { SubmissionStatusContext } from '../context/SubmissionStatusContext/SubmissionStatusContext.js';
import TaxReturnAlert from '../components/Alert/TaxReturnAlert.js';
import MefAlert from '../components/Alert/MefAlert.js';
import useTranslatePIIRedacted from '../hooks/useTranslatePIIRedacted.js';
import CollectionHubDataView from './data-view/CollectionHubDataView.js';
import { getInnerLoopHub, getSectionIndexForScreen, getSectionRouteForScreen } from '../misc/collectionLoopHelpers.js';
import { Path } from '../flow/Path.js';

const isLoop = (sscOrLoop: FlowCollectionLoop | FlowSubSubcategory) => {
  return sscOrLoop.loopName ? true : false;
};

export const renderTaxReturnAlertForDataView = (alertConfig: AlertConfig) => (
  <TaxReturnAlert
    key={buildAlertKey(alertConfig)}
    i18nKey={buildDataViewAlertI18nKey(alertConfig)}
    type={alertConfig.type}
    headingLevel={`h3`}
    collectionId={null}
    internalLink={alertConfig.route}
  />
);

export const renderMefAlertForDataView = (mefAlertConfig: MefAlertConfig) => (
  <MefAlert
    key={buildAlertKey(mefAlertConfig)}
    mefErrorCode={mefAlertConfig.mefErrorCode}
    i18nKey={mefAlertConfig.i18nKey}
    renderLocation='data-view'
    type={mefAlertConfig.type}
    collectionId={null}
    internalLink={mefAlertConfig.route}
  />
);

const extractCategoriesFromPath = (
  path: string
): { subcategoryRoute: string; subSubCategoryRoute: string | undefined } => {
  const pathArray = path.split(`/`);
  let subcategoryRoute = undefined;
  let subSubCategoryRoute = undefined;
  if (pathArray.length < 3) {
    throw new Error(`Malformed dataview path ${path}`);
  }
  subcategoryRoute = `/` + pathArray.slice(0, 3).join(`/`);
  if (pathArray.length >= 4 && pathArray[3] !== ``) {
    subSubCategoryRoute = `/` + pathArray.slice(0, 4).join(`/`);
  }
  return { subcategoryRoute, subSubCategoryRoute };
};

const DataView = () => {
  const flow = useFlow();
  const { factGraph } = useFactGraph();
  const { submissionStatus } = useContext(SubmissionStatusContext);
  const params = useParams();
  const scrolledRef = useRef(false);
  const { t, i18n } = useTranslation();
  const [searchParams] = useSearchParams();
  // Extract the categories from the path
  const { subcategoryRoute, subSubCategoryRoute } = extractCategoriesFromPath(`${params[`*`]}`);

  // Get the whole subcategory object from the flow
  const subcategory = flow.subcategoriesByRoute.get(subcategoryRoute);
  if (!subcategory) {
    throw new Error(`No subcategory route ${subcategoryRoute}`);
  }

  const { hash } = useLocation();

  useEffect(() => {
    if (hash && !scrolledRef.current) {
      const id = hash.replace(`#`, ``);
      const element = document.getElementById(id);
      if (element) {
        // Why the timeout?
        // Chromium based browsers interrupt the scrollIntoView when any other scroll event is triggered.
        // setTimeout set to 0 will add scrollIntoView in the JS event queue then execute.
        setTimeout(() => {
          element.scrollIntoView({ behavior: `instant` });
        }, 0);
        scrolledRef.current = true;
        // Why history manipulation?
        // When a user completes a SubSubSection A, the user is directed towards the dataview page
        // and scrolls to subsubsection A. If then the user clicks edit on SubSubsection B then click back.
        // The page will scroll to SubSubSection A rather than SubSubSection B. Therefore, I removed the hash
        // from the history to prevent that. That works because on the ScreenHeader screen the navigate goBack
        // has the property prevent scroll reset set to true.
        window.history.replaceState(window.history.state, ``, window.location.pathname);
      }
    }
  }, [hash]);

  // Make a map of strings to headings used to summarize alerts
  // These will be passed down into components which will populate the map
  const subSubCategoryRefs = useRef(new Map<string, MutableRefObject<HTMLHeadingElement>>());
  const headingRefs = useRef(new Map<string, MutableRefObject<HTMLHeadingElement>>());

  // CollectionId is only set on a subcategory for about-you and spouse
  const collectionId = subcategory.collectionName
    ? getCollectionId(factGraph, searchParams, subcategory.collectionName)
    : null;

  // COMPLETION OF CATEGORY, SECTIONS AND SCREENS
  // Calculate whether the subcategory is complete
  // First find the completion conditions on the subcategory
  const allSubCatCompletionConditions = Array.isArray(subcategory.completionCondition)
    ? subcategory.completionCondition
    : [subcategory.completionCondition];

  // Then check if any collection item within loops is incomplete or if the collection itself
  // has never been entered.
  const hasIncompleteLoop = subcategory.loops.some((loop: FlowCollectionLoop) => {
    const collectionDone = loop.donePath
      ? factGraph.get(Path.concretePath(loop.donePath, null)).complete &&
        factGraph.get(Path.concretePath(loop.donePath, null)).get
      : false;
    return !collectionDone || hasAtLeastOneIncompleteCollectionItem(loop, factGraph);
  });

  // isComplete is true if all completion conditions for the subcategory are true
  // AND the collections have no incomplete item
  const isComplete =
    allSubCatCompletionConditions.every((condition) => new Condition(condition).evaluate(factGraph, collectionId)) &&
    !hasIncompleteLoop;

  // If the subcategory is not complete, find the next incomplete screen
  // This is so we don't render incomplete sections, screens (and their placeholder facts)
  const nextIncompleteScreen = !isComplete
    ? findFirstIncompleteScreenOfSubcategory(subcategory, factGraph, collectionId)
    : { screen: undefined, id: null };

  // Figure out which section the incomplete screen is in
  const sectionRouteOfIncompleteScreen = nextIncompleteScreen.screen
    ? getSectionRouteForScreen(nextIncompleteScreen.screen, subcategory)
    : undefined;
  const sectionIndexOfIncompleteScreen = sectionRouteOfIncompleteScreen
    ? getSectionIndexForScreen(sectionRouteOfIncompleteScreen, subcategory.subSubcategoriesAndLoops)
    : -1;

  // SETUP OF ALERTS FOR ERRORS AND WARNINGS
  // Find all alerts and assertions
  const hasSubmissionRejectionErrors = submissionStatus && submissionStatus.rejectionCodes.length > 0;
  const mefAlertsConfigs = hasSubmissionRejectionErrors
    ? getMefAlertConfigs(subcategory.screens, collectionId, factGraph, submissionStatus)
    : getEmptyAlertConfigs<MefAlertConfig>();
  const taxReturnAlertConfigs = getTaxReturnAlertConfigs(subcategory.screens, collectionId, factGraph);

  const assertions = useMemo(() => {
    return subcategory.assertions.filter((assertion) => conditionsPass(assertion, factGraph, collectionId));
  }, [collectionId, factGraph, subcategory.assertions]);

  // Move this lower?
  const customDataViewHeading = `dataviews.${subcategory.route}.heading`;
  const headingI18nKey = i18n.exists(customDataViewHeading)
    ? customDataViewHeading
    : `checklist.${subcategory.route}.heading`;

  const sscAssertions = assertions.filter(
    (assertion) => assertion.subSubCategoryRoute !== undefined && conditionsPass(assertion, factGraph, collectionId)
  );

  // COLLECTION LOOPS HANDLING
  // Collection loops can be displayed as a full collection hub
  // or nested (isInner was set), or flattened (autoIterate was set)
  // First find all the collection loops in this category
  const collectionLoops = subcategory.subSubcategoriesAndLoops.filter((sscOrLoop) =>
    isLoop(sscOrLoop)
  ) as FlowCollectionLoop[];
  // eslint-disable-next-line eqeqeq
  const isSubsubcategoryView = subSubCategoryRoute != undefined;

  // Find the full collection hub loops
  let hubLoops = collectionLoops.filter((loop) => !loop.isInner && !loop.autoIterate);
  // or if we are at the subcategory level and find an inner loop
  if (isSubsubcategoryView) {
    hubLoops = collectionLoops
      .filter((loop) => loop.isInner)
      .filter((innerLoops) => innerLoops.fullRoute === subSubCategoryRoute);
  }
  // There can be only one full collection hub max to display
  if (hubLoops.length > 1) {
    throw new Error(`Error: Can only display one collection hub. ${hubLoops.length} found at ${subcategory.route}`);
  }
  // If we found one, set showFullCollectionHub to true
  const showFullCollectionHub = hubLoops.length === 1;
  const hubLoop = hubLoops[0];
  let fullCollectionHub = undefined;
  if (showFullCollectionHub) {
    const availableScreens = subcategory.screens.filter((s) => s.isAvailable(factGraph, collectionId));
    fullCollectionHub = availableScreens.find((screen) => screen.subSubcategoryRoute === hubLoop.fullRoute);
    if (fullCollectionHub === undefined) {
      fullCollectionHub = availableScreens.find((screen) =>
        screen.content.find((contentItem) => {
          return contentItem.componentName === `CollectionItemManager`;
        })
      );
    }
  }

  const showNavButtons = !showFullCollectionHub;

  const redacted = useTranslatePIIRedacted(headingI18nKey, true);

  const showAlert = !fullCollectionHub && !isComplete && nextIncompleteScreen.screen;

  const { summaryErrorSections, summaryWarningSections } = showFullCollectionHub
    ? extractUniqueAlertSummarySectionsForCollectionHub(factGraph, mefAlertsConfigs, taxReturnAlertConfigs, hubLoop)
    : extractUniqueAlertSummarySectionsForDataView(
        mefAlertsConfigs,
        taxReturnAlertConfigs,
        subcategory,
        nextIncompleteScreen
      );
  return (
    <>
      <SectionsAlertAggregator
        summaryErrorSections={summaryErrorSections}
        summaryWarningSections={summaryWarningSections}
        refs={showFullCollectionHub ? headingRefs : subSubCategoryRefs}
      />
      {!showFullCollectionHub && (
        // Show the main header unless full collection hub
        // because the CollectionHubDataView will take care of that.
        <div className='screen__header'>
          <PageTitle redactedTitle={redacted}>
            <Translation i18nKey={`dataviews.reviewColon`} collectionId={null} />
            <Translation i18nKey={headingI18nKey} collectionId={null} />
          </PageTitle>
        </div>
      )}
      {assertions
        .filter((a) => a.subSubCategoryRoute === undefined)
        .map((assertion, index) => {
          const assertionEditScreen = assertion.editRoute ? flow.screensByRoute.get(assertion.editRoute) : null;
          return (
            <Alert key={`${assertion.i18nKey}_${index}`} type='info' headingLevel='h2'>
              <Translation
                i18nKey={assertion.i18nKey}
                collectionId={collectionId}
                components={assertion.editRoute ? { InternalLink: <Link to={assertion.editRoute} /> } : {}}
              />
              &nbsp;
              {assertionEditScreen && (
                <Link to={assertionEditScreen.fullRoute(collectionId)}>{t(`dataviews.edit`)}</Link>
              )}
            </Alert>
          );
        })}
      {!showFullCollectionHub && (
        <>
          <div className='usa-prose'>
            <p>
              <Translation i18nKey={`dataviews.reviewAndEdit`} collectionId={null} />
            </p>
          </div>
        </>
      )}
      {showFullCollectionHub && (
        <CollectionHubDataView
          key={hubLoop.loopName}
          collectionLoop={hubLoop}
          collectionHub={fullCollectionHub}
          variant={`full`}
          subSubCategoryRefs={subSubCategoryRefs}
          headingRefs={headingRefs}
          flow={flow}
        />
      )}

      {!showFullCollectionHub &&
        subcategory.subSubcategoriesAndLoops.map((loopOrSsc, sectionIndex) => {
          if (isLoop(loopOrSsc)) {
            // Handle loops
            const loop = loopOrSsc as FlowCollectionLoop;
            // The innerLoopOffset accounts for the fact that section index is shifted for inner loops
            // as there are 2 entries in subcategoriesAndLoops
            const innerLoopOffset = loop.isInner ? 1 : 0;
            // Don't show sections after the next incomplete screen
            const isAfterNextIncompleteScreen =
              sectionIndexOfIncompleteScreen === -1
                ? false
                : sectionIndex > sectionIndexOfIncompleteScreen + innerLoopOffset;
            if (isAfterNextIncompleteScreen) {
              return null;
            }
            return (
              <CollectionHubDataView
                key={loop.loopName}
                collectionLoop={loop}
                variant={loop.autoIterate ? `flat` : `nested`}
                subSubCategoryRefs={subSubCategoryRefs}
                headingRefs={headingRefs}
                flow={flow}
              />
            );
            // );
          } else if (getInnerLoopHub(loopOrSsc as FlowSubSubcategory) === undefined) {
            // Handle subsubcategories
            const ssc = loopOrSsc as FlowSubSubcategory;
            const assertion = sscAssertions.find((a) => a.subSubCategoryRoute === ssc.fullRoute);
            const subSubCategoryAlertConfigs = getSubSubCategoryAlertConfigs(taxReturnAlertConfigs, ssc.fullRoute);
            const subSubCategoryMefAlertConfigs = getSubSubCategoryAlertConfigs(mefAlertsConfigs, ssc.fullRoute);
            return (
              <SubSubCategory
                ssc={ssc}
                collectionId={collectionId}
                alertConfigs={subSubCategoryAlertConfigs}
                mefAlertConfigs={subSubCategoryMefAlertConfigs}
                refs={subSubCategoryRefs}
                assertion={assertion}
                key={`${ssc.fullRoute}-${collectionId}`}
                isAfterNextIncompleteScreen={
                  sectionIndexOfIncompleteScreen === -1 ? false : sectionIndex > sectionIndexOfIncompleteScreen
                }
                nextIncompleteScreen={nextIncompleteScreen.screen}
                sectionIsComplete={isComplete}
                includesNextIncompleteScreen={sectionIndex === sectionIndexOfIncompleteScreen}
                borderStyle={ssc.borderStyle}
                headingLevel={ssc.headingLevel}
              />
            );
          } else {
            // Inner loops end up having two entries in subcategoriesAndLoops
            // Only one is displayed at this level, this ignores the other one
            return null;
          }
        })}

      <DataViewDynamicNav
        isSignAndSubmit={subcategory.isSignAndSubmit || false}
        showAlert={showAlert}
        showNavButtons={showNavButtons}
        route={
          showAlert && nextIncompleteScreen.screen
            ? nextIncompleteScreen.screen?.fullRoute(nextIncompleteScreen.id)
            : ``
        }
        sectionIsComplete={isComplete}
      />
    </>
  );
};

export default DataView;
