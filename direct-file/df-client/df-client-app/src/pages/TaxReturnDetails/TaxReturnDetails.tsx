import Heading from '../../components/Heading.js';
import DownloadPDFButton from '../../components/DownloadPDFButton/index.js';
import InfoDisplay from '../../components/InfoDisplay.js';
import { ConditionalList } from '../../components/ConditionalList/ConditionalList.js';
import Translation from '../../components/Translation/index.js';
import { useContext } from 'react';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import { FEDERAL_RETURN_STATUS } from '../../constants/taxConstants.js';
import RejectedReturnDetails from './RejectedReturnDetails/RejectedReturnDetails.js';
import { getLatestSubmission, getTaxReturnById } from '../../utils/taxReturnUtils.js';
import { formatAsContentDate } from '../../utils/dateUtils.js';
import { useTranslation } from 'react-i18next';
import { Path } from '../../flow/Path.js';
import useFact from '../../hooks/useFact.js';
import ErroredReturnDetails from './ErroredReturnDetails/ErroredReturnDetails.js';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';

export const TaxReturnDetails = () => {
  const { i18n } = useTranslation(`translation`);
  const { submissionStatus } = useContext(SubmissionStatusContext);
  const { currentTaxReturnId, taxReturns } = useContext(TaxReturnsContext);
  const currentTaxReturn = getTaxReturnById(taxReturns, currentTaxReturnId);
  const [selfSelectPin] = useFact<boolean>(Path.concretePath(`/selfSelectPin`, null));
  const [spouseSelfSelectPin] = useFact<boolean>(Path.concretePath(`/spouseSelfSelectPin`, null));
  if (!(currentTaxReturnId && currentTaxReturn)) return null;

  const numberOfSubmissions = currentTaxReturn.taxReturnSubmissions.length;
  const latestSubmission = getLatestSubmission(currentTaxReturn);
  // TODO: Make sure this isn't messed up by time zones
  const submissionDate = latestSubmission ? new Date(latestSubmission.createdAt) : null;

  const isAccepted = submissionStatus?.status === FEDERAL_RETURN_STATUS.ACCEPTED;
  const isRejected = submissionStatus?.status === FEDERAL_RETURN_STATUS.REJECTED;
  const isError = submissionStatus?.status === FEDERAL_RETURN_STATUS.ERROR;

  const HorizontalRule = () => <hr className='margin-y-4' />;

  return (
    // TODO: Back breadcrumb link thing
    // https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/5125
    <div className='usa-prose'>
      {isRejected && (
        <>
          <RejectedReturnDetails taxReturn={currentTaxReturn} submissionStatus={submissionStatus} />
          <HorizontalRule />
        </>
      )}
      {isError && (
        <>
          <ErroredReturnDetails taxReturn={currentTaxReturn} />
          <HorizontalRule />
        </>
      )}
      <Heading i18nKey='/heading/federal-tax-return' collectionId={null} />
      <p>
        <Translation
          i18nKey='info./info/federal-tax-return/tax-return-id.body'
          collectionId={null}
          context={{ taxReturnId: currentTaxReturn.id }}
        />
      </p>
      <p>
        {submissionDate && (
          <Translation
            i18nKey={
              numberOfSubmissions > 1
                ? `info./info/federal-tax-return/resubmission-date`
                : `info./info/federal-tax-return/submission-date.body`
            }
            collectionId={null}
            context={{ submissionDate: formatAsContentDate(submissionDate, i18n) }}
          />
        )}
      </p>
      {isAccepted && (
        <p>
          <Translation
            i18nKey='info./info/federal-tax-return/accepted-date.body'
            collectionId={null}
            context={{ acceptedDate: formatAsContentDate(new Date(submissionStatus?.createdAt), i18n) }}
          />
        </p>
      )}
      {isRejected && (
        <p>
          <Translation
            i18nKey='info./info/federal-tax-return/rejected-date.body'
            collectionId={null}
            context={{ rejectedDate: formatAsContentDate(new Date(submissionStatus?.createdAt), i18n) }}
          />
        </p>
      )}
      <HorizontalRule />
      <InfoDisplay i18nKey='/info/federal-tax-return/filing-status' collectionId={null} />
      <InfoDisplay i18nKey='/info/federal-tax-return/deductions' collectionId={null} />
      <ConditionalList
        i18nKey='/info/federal-tax-return/deductions'
        collectionId={null}
        items={[
          { itemKey: `educatorDeduction`, conditions: [`/isReceivingEducatorExpensesAdjustment`] },
          { itemKey: `studentLoanDeduction`, conditions: [`/isReceivingStudentLoanInterestAdjustment`] },
          { itemKey: `standardDeduction`, conditions: [`/wantsStandardDeduction`] },
        ]}
      />
      <InfoDisplay i18nKey='/info/federal-tax-return/credits' collectionId={null} />
      <ConditionalList
        i18nKey='/info/federal-tax-return/credits'
        collectionId={null}
        items={[
          { itemKey: `otherDependents`, conditions: [`/isReceivingOdc`] },
          { itemKey: `childTaxCredit`, conditions: [`/isReceivingCtc`] },
          { itemKey: `additionalChild`, conditions: [`/isReceivingActc`] },
          { itemKey: `earnedIncome`, conditions: [`/isReceivingEitc`] },
        ]}
      />
      <HorizontalRule />
      {selfSelectPin && (
        <div className='margin-y-2'>
          <InfoDisplay i18nKey='/info/federal-tax-return/self-select-pin' collectionId={null} />
        </div>
      )}
      {spouseSelfSelectPin && (
        <div className='margin-y-2'>
          <InfoDisplay i18nKey='/info/federal-tax-return/spouse-select-pin' collectionId={null} />
        </div>
      )}
      <DownloadPDFButton taxId={currentTaxReturnId} i18nKey='button.downloadPDF' />
      <InfoDisplay i18nKey='/info/federal-tax-return/irs-online-account' collectionId={null} />
    </div>
  );
};
