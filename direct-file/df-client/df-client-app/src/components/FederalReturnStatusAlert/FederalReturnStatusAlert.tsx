import { Alert } from '@trussworks/react-uswds';
import { useTranslation } from 'react-i18next';
import { TaxReturn, TaxReturnSubmissionStatus } from '../../types/core.js';
import { formatAsContentDate } from '../../utils/dateUtils.js';
import { getLatestSubmission } from '../../utils/taxReturnUtils.js';
import { areAnyRejectionsNotFixable } from '../../utils/submissionStatusUtils.js';
import { HTMLAttributes } from 'react';
import { assertNever } from 'assert-never';
import ContentDisplay from '../ContentDisplay/ContentDisplay.js';
import useFact from '../../hooks/useFact.js';
import { Path } from '../../flow/Path.js';

type DivProps = Omit<HTMLAttributes<HTMLDivElement>, `children`>;

export type FederalReturnStatusAlertProps = {
  taxReturn: TaxReturn;
  submissionStatus: TaxReturnSubmissionStatus;
} & DivProps;

const FederalReturnStatusAlert = ({ taxReturn, submissionStatus, ...divProps }: FederalReturnStatusAlertProps) => {
  switch (submissionStatus.status) {
    case `Pending`: {
      return <PendingReturnStatusAlert taxReturn={taxReturn} {...divProps} />;
    }
    case `Rejected`: {
      return <RejectedReturnStatusAlert taxReturn={taxReturn} submissionStatus={submissionStatus} {...divProps} />;
    }
    case `Accepted`: {
      return <AcceptedReturnStatusAlert submissionStatus={submissionStatus} {...divProps} />;
    }
    case `Error`: {
      return <ErrorReturnStatusAlert taxReturn={taxReturn} {...divProps} />;
    }
    default: {
      assertNever(submissionStatus.status);
    }
  }
};

export default FederalReturnStatusAlert;

type PendingReturnStatusAlertProps = {
  taxReturn: TaxReturn;
} & DivProps;
const PendingReturnStatusAlert = ({ taxReturn, ...divProps }: PendingReturnStatusAlertProps) => {
  const { t, i18n } = useTranslation(`translation`);

  const latestSubmission = getLatestSubmission(taxReturn);
  const latestSubmitDate = latestSubmission && new Date(latestSubmission.createdAt);
  const wasResubmitted = taxReturn.taxReturnSubmissions.length > 1;

  return (
    <Alert
      type='info'
      headingLevel='h3'
      heading={t(
        wasResubmitted
          ? `federalReturnStatusAlert.pending.resubmitted.heading`
          : `federalReturnStatusAlert.pending.heading`,
        {
          taxYear: taxReturn.taxYear,
          submittedDate: latestSubmitDate ? formatAsContentDate(latestSubmitDate, i18n) : ``,
        }
      )}
      {...divProps}
    >
      <ContentDisplay
        i18nKey={wasResubmitted ? `federalReturnStatusAlert.pending.resubmitted` : `federalReturnStatusAlert.pending`}
        collectionId={null}
      />
    </Alert>
  );
};

type RejectedReturnStatusAlertProps = {
  taxReturn: TaxReturn;
  submissionStatus: TaxReturnSubmissionStatus;
} & DivProps;
const RejectedReturnStatusAlert = ({ taxReturn, submissionStatus, ...divProps }: RejectedReturnStatusAlertProps) => {
  const { t, i18n } = useTranslation(`translation`);
  const [isAfterResubmissionDeadline] = useFact<boolean>(Path.concretePath(`/isAfterResubmissionDeadline`, null));

  const rejectedDate = new Date(submissionStatus.createdAt);

  const notFixableInDF = areAnyRejectionsNotFixable(submissionStatus.rejectionCodes);

  return (
    <>
      {isAfterResubmissionDeadline ? (
        <Alert
          type='error'
          headingLevel='h3'
          heading={t(`federalReturnStatusAlert.rejected.postResubmissionDeadlineHeading`, {
            rejectedDate: formatAsContentDate(rejectedDate, i18n),
          })}
          {...divProps}
        />
      ) : (
        <Alert
          type='error'
          headingLevel='h3'
          heading={t(`federalReturnStatusAlert.rejected.heading`, {
            taxYear: taxReturn.taxYear,
            rejectedDate: formatAsContentDate(rejectedDate, i18n),
          })}
          {...divProps}
        >
          <ContentDisplay
            i18nKey={
              notFixableInDF
                ? `federalReturnStatusAlert.rejected.cannotFixInDirectFile`
                : `federalReturnStatusAlert.rejected.canFixInDirectFile`
            }
            collectionId={null}
            context={{
              count: submissionStatus.rejectionCodes.length,
            }}
          />
        </Alert>
      )}
    </>
  );
};

type AcceptedReturnStatusAlertProps = {
  submissionStatus: TaxReturnSubmissionStatus;
} & DivProps;
const AcceptedReturnStatusAlert = ({ submissionStatus, ...divProps }: AcceptedReturnStatusAlertProps) => {
  const { t, i18n } = useTranslation(`translation`);

  // Note: For an Accepted submissionStatus, `createdAt` most nearly corresponds to when DF was made aware that the
  //       return was accepted. It is not the exact moment in time MeF made the judgement, however.
  const acceptedDate = submissionStatus.createdAt;

  return (
    <Alert
      type='success'
      headingLevel='h3'
      heading={t(`federalReturnStatusAlert.accepted.heading`, {
        acceptedDate: formatAsContentDate(new Date(acceptedDate), i18n),
      })}
      {...divProps}
    />
  );
};

// Note: This component has not been built with formally approved designs or content, and is a placeholder for MVP
type ErrorReturnStatusAlertProps = {
  taxReturn: TaxReturn;
} & DivProps;
const ErrorReturnStatusAlert = ({ taxReturn, ...divProps }: ErrorReturnStatusAlertProps) => {
  const { t } = useTranslation(`translation`);
  const [isBeforeFederalTaxDeadline] = useFact<boolean>(Path.concretePath(`/isBeforeFederalTaxDeadline`, null));

  return (
    <>
      {isBeforeFederalTaxDeadline ? (
        <Alert
          type='error'
          headingLevel='h3'
          heading={t(`federalReturnStatusAlert.error.heading`, { taxYear: taxReturn.taxYear })}
          {...divProps}
        >
          <ContentDisplay i18nKey={`federalReturnStatusAlert.error`} collectionId={null} />
        </Alert>
      ) : (
        <Alert
          type='error'
          headingLevel='h3'
          heading={t(`federalReturnStatusAlert.error.headingPostDeadline`)}
          {...divProps}
        />
      )}
    </>
  );
};
