import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { useFlow } from '../../flow/flowConfig.js';
import { Link, Navigate, useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import {
  MefAlertConfig,
  extractUniqueAlertSummarySectionsForDataView,
  filterAlertsBySubSubCategory,
  getEmptyAlertConfigs,
  getMefAlertConfigs,
  getTaxReturnAlertConfigs,
} from '../../misc/aggregatedAlertHelpers.js';
import { SubSubCategory } from './SubSubCategory.js';
import { RefObject, useCallback, useContext, useRef } from 'react';
import { Alert } from '@trussworks/react-uswds';
import { findFirstIncompleteScreenOfLoop } from '../../flow/flowHelpers.js';
import { ConcretePath } from '@irs/js-factgraph-scala';
import Translation from '../../components/Translation/index.js';
import styles from '../../components/Screen.module.scss';
import { useSaveAndPersistIfPossible } from '../../hooks/useSaveAndPersistIfPossible.js';
import { Condition } from '../../flow/Condition.js';
import { conditionsPass } from '../../utils/condition.js';
import Assertion from './Assertion.js';
import { useTranslation } from 'react-i18next';
import Result from './ResultAssertion.js';
import useTranslateWithFacts from '../../hooks/useTranslateWithFacts.js';
import ConfirmationModal from '../../components/ConfirmationModal/ConfirmationModal.js';
import { DataViewButton } from './DataViewButton.js';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import SectionsAlertAggregator from '../../components/SectionsAlertAggregator/SectionsAlertAggregator.js';
import PageTitle from '../../components/PageTitle/PageTitle.js';
import { isValidCollectionInURL } from '../BaseScreen.js';
import { handleRefFromRoute } from '../../components/SummaryAlert/summaryHelpers.js';
import { useIsReturnEditable } from '../../hooks/useIsReturnEditable.js';
import { getCollectionItemThatNeedsManualReview, routeToScreenWithItemId } from '../navUtils.js';
import { AbsolutePath } from '../../fact-dictionary/Path.js';
import { Path } from '../../flow/Path.js';

export const getCollectionDataviewTitleKey = (loopName: string) => {
  return loopName === `/familyAndHousehold` ? `dataviews.personDetails` : `dataviews.incomeDetails`;
};

const CollectionItemDataView = () => {
  const [searchParams] = useSearchParams();
  const flow = useFlow();
  const { factGraph } = useFactGraph();
  const { submissionStatus } = useContext(SubmissionStatusContext);
  const { isReturnEditable } = useIsReturnEditable();
  const { state } = useLocation();
  const params = useParams();
  const loopName = params.loopName;
  const collectionId = params.collectionId;
  const navigate = useNavigate();
  const { t, i18n } = useTranslation();
  const { contextHasData } = useTranslateWithFacts(factGraph, collectionId);
  const childCategoryRefs = useRef(new Map<string, React.MutableRefObject<HTMLHeadingElement>>());

  if (!loopName || !collectionId) {
    throw new Error(`Could not load collection data view -- missing loopName or collectionId`);
  }
  const collectionItemTitle = t(getCollectionDataviewTitleKey(loopName));
  const loop = flow.collectionLoopsByName.get(loopName);
  if (!loop) {
    throw new Error(`No loop for loopname ${loopName}`);
  }

  const collectionItemIsImportedMaybeValue =
    loop.loopName && collectionId && loop.isImportedFactPath
      ? factGraph.get(Path.concretePath(loop.isImportedFactPath as AbsolutePath, collectionId))
      : null;
  const collectionItemIsImported = collectionItemIsImportedMaybeValue?.hasValue
    ? collectionItemIsImportedMaybeValue.get === true
    : false;

  if (collectionItemIsImported && loop.importedFlowDonePath) {
    factGraph.set(Path.concretePath(loop.importedFlowDonePath, collectionId), true);
    // eslint-disable-next-line df-rules/no-factgraph-save
    factGraph.save();
  }
  const collectionHubRoute = `/data-view${loop.fullRoute}`;
  const routeNext = flow.collectionLoopsByName.get(loopName)?.importedRouteOverride || collectionHubRoute;

  const saveAndPersistIfPossible = useSaveAndPersistIfPossible();

  const handleDelete = useCallback(async () => {
    factGraph.delete(`${loop.collectionName}/#${collectionId}` as ConcretePath);
    const persistResult = await saveAndPersistIfPossible();
    const hasPersistError = persistResult.hasPersistError;
    if (!hasPersistError) {
      navigate(`/data-view${loop.fullRoute}`);
    }
  }, [factGraph, loop.collectionName, loop.fullRoute, collectionId, saveAndPersistIfPossible, navigate]);

  if (!isValidCollectionInURL(factGraph, searchParams, loop.collectionName as ConcretePath, collectionId)) {
    // if we're at an invalid URL, navigate to a 404 page
    return <Navigate to='../not-found' replace />;
  }
  // if there's no collectionItemCompletedCondition, the only thing we'll care about is whether
  // there's incomplete data on the screens.
  const isComplete = loop.collectionItemCompletedCondition
    ? new Condition(loop.collectionItemCompletedCondition).evaluate(factGraph, collectionId)
    : false;

  const nextIncompleteScreen = !isComplete
    ? { screen: findFirstIncompleteScreenOfLoop(loop, factGraph, collectionId), id: collectionId }
    : { screen: undefined, id: null };

  const alertConfigs = getTaxReturnAlertConfigs(loop.screens, collectionId, factGraph);
  const mefAlertConfigs =
    (submissionStatus && getMefAlertConfigs(loop.screens, collectionId, factGraph, submissionStatus)) ||
    getEmptyAlertConfigs<MefAlertConfig>();

  const { summaryWarningSections, summaryErrorSections } = extractUniqueAlertSummarySectionsForDataView(
    mefAlertConfigs,
    alertConfigs,
    loop,
    nextIncompleteScreen
  );
  const matchingDataViewSection = loop.dataViewSections?.find((dvs) => conditionsPass(dvs, factGraph, collectionId));
  const assertion = matchingDataViewSection?.itemAssertions?.find(
    (a) => conditionsPass(a, factGraph, collectionId) && isComplete
  );
  const assertionEditScreen = loop.subSubcategories
    .find((ssc) => ssc.routeSuffix === assertion?.subSubCategoryToEdit)
    ?.screens.find((s) => s.isAvailable(factGraph, collectionId));

  const outcome = matchingDataViewSection?.itemAssertions?.find(
    (a) => conditionsPass(a, factGraph, collectionId) && a.outcomeI18nKey
  );
  const outcomeReviewScreen = loop.subSubcategories
    .find((ssc) => ssc.routeSuffix === outcome?.outcomeReviewRoute)
    ?.screens.find((s) => s.isAvailable(factGraph, collectionId));

  const i18nKeys = (leafKey: string) => {
    return [`fields.${loop.loopName}.${leafKey}`, `fields.${loop.collectionName}.collectionListing.${leafKey}`];
  };
  const primaryItemHeadingKey = i18nKeys(`itemHeading1`);
  const secondaryItemHeadingKey = i18nKeys(`itemHeading2`);
  const hasPrimaryItemLabel1 = i18nKeys(`label1`).some((key) => i18n.exists(key));
  const hasItemDescription = i18nKeys(`itemDescription`).some((key) => i18n.exists(key));
  const hasPrimaryItemHeading = i18n.exists(primaryItemHeadingKey) && contextHasData(primaryItemHeadingKey);
  const isReviewMode = searchParams.get(`reviewMode`);

  const collectionItemToReview = loop.importedFlowDonePath
    ? getCollectionItemThatNeedsManualReview(factGraph, loop.loopName, loop.importedFlowDonePath)
    : null;

  const nextScreenToReview =
    state && collectionItemToReview && loop.importedFlowStartRoute
      ? routeToScreenWithItemId(loop.importedFlowStartRoute, loop.loopName, collectionItemToReview)
      : collectionHubRoute;

  return (
    <div>
      <SectionsAlertAggregator
        summaryErrorSections={summaryErrorSections}
        summaryWarningSections={summaryWarningSections}
        refs={childCategoryRefs}
        collectionId={collectionId}
        collectionName={loop.collectionName}
      />
      <div className='screen__header'>
        <div className='tablet:display-flex flex-justify flex-align-baseline'>
          <PageTitle redactedTitle={collectionItemTitle}>
            <Translation i18nKey={`dataviews.reviewColon`} collectionId={null} />
            <Translation
              i18nKey={hasPrimaryItemHeading ? primaryItemHeadingKey : secondaryItemHeadingKey}
              collectionId={collectionId}
            />
          </PageTitle>
          {isReturnEditable && (
            <ConfirmationModal
              i18nKey={`fields.${loop.collectionName}.${
                collectionItemIsImported ? `importedDeleteControl` : `deleteControl`
              }`}
              handleConfirm={handleDelete}
              collectionId={collectionId}
              destructiveAction
              icon='Delete'
              modalOpenerClasses='text-no-wrap'
            />
          )}
        </div>
        {hasItemDescription && (
          <p>
            <Translation i18nKey={i18nKeys(`itemDescription`)} collectionId={collectionId} />
          </p>
        )}
        {hasPrimaryItemHeading && hasPrimaryItemLabel1 && (
          <p>
            <Translation i18nKey={i18nKeys(`label1`)} collectionId={collectionId} />
          </p>
        )}
        {assertion && assertion.resultI18nKey && (
          <p>
            <Translation i18nKey={assertion.resultI18nKey} collectionId={collectionId} />
          </p>
        )}
      </div>
      {assertion && assertion.collectionItemI18nKey && (
        <>
          <Assertion
            type={assertion.type}
            i18nKey={assertion.collectionItemI18nKey}
            collectionId={collectionId}
            editRoute={assertionEditScreen?.fullRoute(collectionId)}
          />
        </>
      )}

      <div>
        {loop.subSubcategories.map((ssc, index) => {
          const sscAlertsConfigs = filterAlertsBySubSubCategory(alertConfigs, ssc.fullRoute);
          const sscMefAlertsConfigs = mefAlertConfigs && filterAlertsBySubSubCategory(mefAlertConfigs, ssc.fullRoute);
          const indexOfNextIncompleteScreen = loop.subSubcategories.findIndex((ssc) => {
            return nextIncompleteScreen.screen
              ? ssc.fullRoute === nextIncompleteScreen.screen.subSubcategoryRoute
              : false;
          });
          return (
            <SubSubCategory
              key={ssc.fullRoute}
              ssc={ssc}
              collectionId={collectionId}
              alertConfigs={sscAlertsConfigs}
              mefAlertConfigs={sscMefAlertsConfigs}
              refs={childCategoryRefs}
              nextIncompleteScreen={nextIncompleteScreen.screen}
              isAfterNextIncompleteScreen={
                indexOfNextIncompleteScreen === -1 ? false : index > indexOfNextIncompleteScreen
              }
              includesNextIncompleteScreen={index === indexOfNextIncompleteScreen}
              sectionIsComplete={isComplete}
            />
          );
        })}
      </div>
      {outcome?.outcomeI18nKey && (
        <Result
          asHeading
          i18nKey={outcome.outcomeI18nKey}
          reviewRoute={outcomeReviewScreen?.fullRoute(collectionId)}
          collectionId={collectionId}
        />
      )}

      {!isComplete && nextIncompleteScreen.screen && (
        <div className={styles.collectionItemDataView}>
          <Alert className='margin-y-3 ' type='error' headingLevel='h3' validation>
            <p className='margin-y-0'>
              <Translation i18nKey={`dataviews.incomplete`} collectionId={collectionId} />
              &nbsp;
              <Link
                ref={
                  handleRefFromRoute(
                    nextIncompleteScreen.screen.fullRoute(collectionId, { reviewMode: false }),
                    childCategoryRefs
                  ) as RefObject<HTMLAnchorElement>
                }
                to={nextIncompleteScreen.screen.fullRoute(collectionId, { reviewMode: false })}
              >
                {t(`dataviews.resume`)}
              </Link>
            </p>
          </Alert>
        </div>
      )}
      <div className='padding-top-4'>
        {collectionItemIsImported ? (
          <DataViewButton
            route={collectionItemToReview ? nextScreenToReview : routeNext}
            isOutline={false}
            i18nKey={`button.continue`}
          />
        ) : (
          <DataViewButton
            route={collectionHubRoute}
            isOutline={isComplete ? false : true}
            i18nKey={isReviewMode || !isComplete ? `fields.${loop.collectionName}.controls.back` : `button.continue`}
          />
        )}
      </div>
    </div>
  );
};

export default CollectionItemDataView;
