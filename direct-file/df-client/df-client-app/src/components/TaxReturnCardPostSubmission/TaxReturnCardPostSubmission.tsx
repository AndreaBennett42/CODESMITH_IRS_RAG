import { FC, useContext, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { Grid } from '@trussworks/react-uswds';

import { TaxReturn } from '../../types/core.js';
import { FEDERAL_RETURN_STATUS } from '../../constants/taxConstants.js';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import DownloadPDFButton from '../DownloadPDFButton/index.js';
import Translation from '../Translation/index.js';
import { Path } from '../../flow/Path.js';
import Subheading from '../Subheading.js';
import InfoDisplay from '../InfoDisplay.js';
import useFact from '../../hooks/useFact.js';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import {
  isBeforeEndOfPerfectionDeadline,
  isBetweenFilingDeadlineAndPerfectionDeadline,
  isPostFederalFilingDeadline,
} from '../../utils/dateUtils.js';
import { areAnyRejectionsNotFixable } from '../../utils/submissionStatusUtils.js';
import StateTaxesCard from '../StateTaxesCard/StateTaxesCard.js';
import { Enum as EnumValue } from '@irs/js-factgraph-scala';
import { StateOrProvince } from '../../types/StateOrProvince.js';
import FederalReturnStatusAlert from '../FederalReturnStatusAlert/FederalReturnStatusAlert.js';
import PaperPathStatusAlert from '../PaperPathStatusAlert/PaperPathStatusAlert.js';
import { usePollForSubmissionStatus } from '../../hooks/usePollForSubmissionStatus.js';
import useFetchStateProfile from '../../hooks/useFetchStateProfile.js';
import { QuestionsReminderCard } from '../TaxReturnCard/SimpleReminderTaxReturnCard.js';
import IntroContent from '../IntroContent/IntroContent.js';
import { getLatestSubmission } from '../../utils/taxReturnUtils.js';

export interface TaxReturnPostSubmissionProps {
  taxReturn: TaxReturn;
}

export const TaxReturnCardPostSubmission: FC<TaxReturnPostSubmissionProps> = ({ taxReturn }) => {
  const { t } = useTranslation(`translation`);
  const now = new Date();
  const {
    submissionStatus: retrievedSubmissionStatus,
    isFetching: isFetchingStatus,
    fetchSuccess,
  } = useContext(SubmissionStatusContext);
  const latestSubmission = getLatestSubmission(taxReturn);

  const [owesBalance] = useFact<boolean>(Path.concretePath(`/owesBalance`, null));
  const [payViaAch] = useFact<boolean>(Path.concretePath(`/payViaAch`, null));

  const [dueRefund] = useFact<boolean>(Path.concretePath(`/dueRefund`, null));
  const [refundViaAch] = useFact<boolean>(Path.concretePath(`/refundViaAch`, null));

  const [filingStateOrProvince] = useFact<EnumValue>(Path.concretePath(`/filingStateOrProvince`, null));
  const filingStateCode = filingStateOrProvince?.getValue().toUpperCase() as StateOrProvince;
  const [stateCanTransferData] = useFact<boolean>(Path.concretePath(`/stateCanTransferData`, null));
  const [isPaperPath] = useFact<boolean>(Path.concretePath(`/isPaperPath`, null));

  // TODO: The facts below are hardcoded and are not to be changed (at the time of writing).
  //       That said, they will appear to conflict with isAfterTaxDay, and might change in the future.
  const isAfterTaxDay = isPostFederalFilingDeadline(now);
  const isBeforeEndOfPerfectionPeriod = isBeforeEndOfPerfectionDeadline(now);
  const [isBeforeFederalTaxDeadline] = useFact<boolean>(Path.concretePath(`/isBeforeFederalTaxDeadline`, null));
  const [isAfterResubmissionDeadline] = useFact<boolean>(Path.concretePath(`/isAfterResubmissionDeadline`, null));

  const latestSubmissionWasBetweenFilingDeadlineAndPerfectionDeadline = latestSubmission
    ? isBetweenFilingDeadlineAndPerfectionDeadline(new Date(latestSubmission.createdAt))
    : false;

  const { stateProfile, isFetching: loadingStateProfile } = useFetchStateProfile();
  const isFetching = isFetchingStatus || loadingStateProfile;

  const HorizontalRule = () => <hr className='margin-y-4' />;

  // poll for status
  usePollForSubmissionStatus(taxReturn);

  const submissionStatus = useMemo(() => {
    // To avoid flickering a pending status before the first submission status value is returned,
    // give it a chance to try
    const hasFetchedStatusAtLeastOnce = !isFetchingStatus && fetchSuccess;
    if (hasFetchedStatusAtLeastOnce) {
      return retrievedSubmissionStatus;
    }
  }, [isFetchingStatus, fetchSuccess, retrievedSubmissionStatus]);

  const returnSubmissionFailed =
    submissionStatus?.status === FEDERAL_RETURN_STATUS.ERROR && submissionStatus.rejectionCodes.length === 0;
  const returnWasRejected =
    submissionStatus?.status === FEDERAL_RETURN_STATUS.REJECTED && submissionStatus.rejectionCodes.length > 0;
  const hasUnfixableRejection = submissionStatus ? areAnyRejectionsNotFixable(submissionStatus.rejectionCodes) : false;
  const returnRejectedButFixable = returnWasRejected && !hasUnfixableRejection;
  const returnRejectedAndNotFixable = returnWasRejected && hasUnfixableRejection;

  const stateTaxActionMayBeNeeded = filingStateCode && stateProfile;

  const returnWasAccepted = submissionStatus?.status === FEDERAL_RETURN_STATUS.ACCEPTED;

  // TODO: Localize this once we start localizing
  const stateTaxesId = `state-taxes`;

  return (
    <>
      <Grid col={12} className='border-base-lighter border-2px'>
        {/*
          TODO: Handle scenario where load is unsuccessful
          https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/3539
        */}

        <h2 className='margin-0  bg-base-lightest padding-y-2 padding-x-3'>
          {t(`taxReturnCard.header`, taxReturn as { taxYear: number })}
        </h2>
        {!isFetching && fetchSuccess && (
          <div className='padding-205'>
            {submissionStatus && <FederalReturnStatusAlert taxReturn={taxReturn} submissionStatus={submissionStatus} />}
            {isPaperPath && <PaperPathStatusAlert />}
            {!returnWasRejected && (
              <p>
                <CommonLinkRenderer url='/federal-tax-return'>
                  {t(`taxReturnCard.federalTaxReturnLinkText`)}
                </CommonLinkRenderer>
              </p>
            )}
            <p>
              <Translation i18nKey='taxReturnCard.taxReturnId' collectionId={null} context={taxReturn} />
            </p>
            {!returnWasRejected && <DownloadPDFButton taxId={taxReturn.id} i18nKey='button.downloadPDF' />}
            {(owesBalance || dueRefund) && !returnWasRejected && (
              <>
                <HorizontalRule />
                <Subheading i18nKey='subheadings./subheading/complete/sign-and-submit/next' collectionId={null} />
              </>
            )}
            {owesBalance && !returnWasRejected && (
              <InfoDisplay
                i18nKey={`/info/complete/sign-and-submit/tax-owed-${
                  payViaAch
                    ? `pay-now${returnWasAccepted ? `` : `-pending`}`
                    : `pay-later${isAfterTaxDay ? `-after-tax-day` : ``}`
                }`}
                collectionId={null}
              />
            )}
            {dueRefund && !returnWasRejected && (
              <>
                <InfoDisplay
                  i18nKey={`/info/complete/sign-and-submit/refund-info-${refundViaAch ? `direct-deposit` : `check`}`}
                  collectionId={null}
                />
                <InfoDisplay
                  i18nKey={`/info/complete/sign-and-submit/refund-amount-explainer${
                    returnWasAccepted || !refundViaAch ? `-with-wheres-my-refund` : ``
                  }`}
                  collectionId={null}
                />
              </>
            )}
            {returnRejectedButFixable &&
              (isAfterResubmissionDeadline ? (
                <>
                  <HorizontalRule />
                  <h3>{t(`taxReturnCard.status.rejected.remediation.postResubmissionDeadlineHeading.heading`)}</h3>
                  <p className='usa-prose'>
                    <IntroContent
                      i18nKey='taxReturnCard.status.rejected.remediation.postResubmissionDeadlineHeading'
                      collectionId={null}
                    />
                  </p>
                  <div className='screen__actions'>
                    <CommonLinkRenderer className='usa-button width-full' url='/federal-tax-return'>
                      {t(`taxReturnCard.status.rejected.remediation.postResubmissionDeadlineHeading.detailsButton`)}
                    </CommonLinkRenderer>
                    <CommonLinkRenderer
                      className='usa-button usa-button--outline width-full'
                      url={t(`commonUrls.otherWaysToFile`)}
                    >
                      {t(`taxReturnCard.otherWaysToFile`)}
                    </CommonLinkRenderer>
                  </div>
                </>
              ) : (
                <>
                  <HorizontalRule />
                  <h3>{t(`taxReturnCard.status.rejected.remediation.heading`)}</h3>
                  <p className='usa-prose'>
                    <IntroContent
                      i18nKey={`taxReturnCard.status.rejected.remediation.${
                        isBeforeEndOfPerfectionPeriod && !latestSubmissionWasBetweenFilingDeadlineAndPerfectionDeadline
                          ? `beforeEndOfPerfectionPeriod`
                          : `asSoonAsPossible`
                      }`}
                      collectionId={null}
                    />
                  </p>
                  <div className='screen__actions'>
                    <CommonLinkRenderer className='usa-button width-full' url='/federal-tax-return'>
                      {t(`taxReturnCard.status.rejected.remediation.reviewErrorsButton`, {
                        count: submissionStatus.rejectionCodes.length,
                      })}
                    </CommonLinkRenderer>
                  </div>
                </>
              ))}
            {returnSubmissionFailed && (
              <>
                <h3>{t(`taxReturnCard.status.error.heading`)}</h3>
                <p className='usa-prose'>{t(`taxReturnCard.status.error.body`)}</p>
                <CommonLinkRenderer className='usa-button width-full' url={t(`commonUrls.otherWaysToFile`)}>
                  {t(`taxReturnCard.otherWaysToFile`)}
                </CommonLinkRenderer>
              </>
            )}
            {returnRejectedAndNotFixable && (
              <>
                <p className='usa-prose'>
                  {isBeforeFederalTaxDeadline ? (
                    <>
                      <h3>{t(`taxReturnCard.status.rejected.alternativeFiling.heading`)}</h3>
                      <IntroContent collectionId={null} i18nKey={`taxReturnCard.status.rejected.alternativeFiling`} />
                    </>
                  ) : (
                    <>
                      <p>
                        <Translation i18nKey='taxReturnCard.whatToDoNext' collectionId={null} />
                      </p>
                      <p>
                        <Translation i18nKey='taxReturnCard.status.rejected.fixErrors' collectionId={null} />
                      </p>
                    </>
                  )}
                </p>
                <div className='screen__actions'>
                  <CommonLinkRenderer className='usa-button width-full' url={t(`commonUrls.otherWaysToFile`)}>
                    {t(`taxReturnCard.otherWaysToFile`)}
                  </CommonLinkRenderer>
                  <CommonLinkRenderer className='usa-button usa-button--outline width-full' url='/federal-tax-return'>
                    {t(`taxReturnCard.status.rejected.alternativeFiling.taxReturnDetailsButton`)}
                  </CommonLinkRenderer>
                </div>
              </>
            )}
          </div>
        )}
      </Grid>

      {/* There are scenarios where the federal return just needs to be submitted to start filing state taxes */}
      {stateTaxActionMayBeNeeded && !returnSubmissionFailed && (
        <StateTaxesCard
          id={stateTaxesId}
          taxYear={taxReturn.taxYear}
          stateProfile={stateProfile}
          stateCanTransferData={!!stateCanTransferData}
          returnWasRejected={returnWasRejected}
        />
      )}

      {(returnWasAccepted ||
        isAfterResubmissionDeadline ||
        ((returnSubmissionFailed || returnRejectedAndNotFixable) && !isBeforeFederalTaxDeadline)) && (
        <QuestionsReminderCard />
      )}
    </>
  );
};
