import { useTranslation } from 'react-i18next';
import { TaxReturn, TaxReturnSubmissionStatus } from '../../../types/core.js';
import { FILING_DEADLINE } from '../../../constants/taxConstants.js';
import { formatAsContentDate } from '../../../utils/dateUtils.js';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import { areAnyRejectionsNotFixable } from '../../../utils/submissionStatusUtils.js';
import InternalLink from '../../../components/InternalLink/index.js';
import { Accordion } from '@trussworks/react-uswds';
import ContentDisplay from '../../../components/ContentDisplay/ContentDisplay.js';
import Translation from '../../../components/Translation/Translation.js';
import DFModal from '../../../components/HelperText/DFModal.js';
import DownloadPDFButton from '../../../components/DownloadPDFButton/index.js';
import IntroContent from '../../../components/IntroContent/IntroContent.js';
import useFact from '../../../hooks/useFact.js';
import { Path } from '../../../flow/Path.js';
import { useCommonUrls } from '../../../hooks/useCommonUrls.js';
import { useContext } from 'react';
import { TaxReturnsContext } from '../../../context/TaxReturnsContext.js';

export type RejectedReturnDetailsProps = {
  taxReturn: TaxReturn;
  submissionStatus: TaxReturnSubmissionStatus;
};

type ReturnDetailsProps = {
  mefErrorCodes: string[];
  taxYear: number;
};

const internalLinkList = [
  {
    key: `SignReturnDataView`,
    url: `/data-view/flow/complete/sign-and-submit`,
  },
  {
    key: `AboutYou`,
    url: `/flow/you-and-your-family/about-you/about-you-basic-info`,
  },
  {
    key: `AboutYouSpouse`,
    url: `/flow/you-and-your-family/spouse/spouse-mfj-basic-info`,
  },
  {
    key: `DependentsDataView`,
    url: `/data-view/flow/you-and-your-family/dependents`,
  },
  {
    key: `AboutYouDataView`,
    url: `/data-view/flow/you-and-your-family/about-you`,
  },
  {
    key: `SpouseDataView`,
    url: `/data-view/flow/you-and-your-family/spouse`,
  },
  {
    key: `JobsDataView`,
    url: `/data-view/flow/income/jobs`,
  },
  {
    key: `CreditsDataView`,
    url: `/data-view/flow/credits-and-deductions/credits`,
  },
];

function buildInternalLinks(i18nKey: string, links: { key: string; url: string }[]) {
  const allLinks: {
    [key: string]: JSX.Element;
  } = {};
  links.forEach((link) => {
    allLinks[`${link.key}InternalLink`] = (
      <InternalLink collectionId={null} i18nKey={i18nKey} route={`${link.url}?reviewMode=true`} />
    );
  });
  return allLinks;
}

const FixableReturnDetails = ({ mefErrorCodes }: ReturnDetailsProps) => {
  const { t, i18n } = useTranslation(`translation`);
  const { currentTaxReturnId } = useContext(TaxReturnsContext);
  const [isAfterResubmissionDeadline] = useFact<boolean>(Path.concretePath(`/isAfterResubmissionDeadline`, null));
  const deadlinePrefix = isAfterResubmissionDeadline ? `postResubmitDeadline` : `preResubmitDeadline`;
  const singleErrorKeyPrefix = `taxReturnDetails.rejectedReturnDetails.fixable.singleErrorExplanation`;
  const fixableReturnKey = `${singleErrorKeyPrefix}.${mefErrorCodes[0]}.${deadlinePrefix}`;
  const hasOneMefErrorWithContent = mefErrorCodes.length === 1 && i18n.exists(fixableReturnKey);
  const commonUrls = useCommonUrls(t);
  const internalLinks = buildInternalLinks(`${fixableReturnKey}.howToFix`, internalLinkList);

  return (
    <>
      {hasOneMefErrorWithContent ? (
        <>
          <h2>{t(`taxReturnDetails.rejectedReturnDetails.errorCode`, { errorCode: mefErrorCodes[0] })}</h2>
          <ContentDisplay i18nKey={`${fixableReturnKey}.errorInPlainLanguage`} />
          {!isAfterResubmissionDeadline && <h2>{t(`taxReturnDetails.rejectedReturnDetails.howToFixError`)}</h2>}
          <ContentDisplay
            i18nKey={`${fixableReturnKey}.howToFix`}
            allowedTags={[`p`, `h2`, `ul`, `li`]}
            additionalComponents={{
              ...commonUrls,
              ...internalLinks,
              Modal: (
                <DFModal
                  collectionId={null}
                  i18nKey={`taxReturnDetails.rejectedReturnDetails.paperReturnModal.${deadlinePrefix}`}
                  context={{
                    filingDeadline: formatAsContentDate(FILING_DEADLINE, i18n),
                  }}
                  additionalComponents={
                    currentTaxReturnId
                      ? {
                          InlinePDFButton: (
                            <DownloadPDFButton taxId={currentTaxReturnId} i18nKey={`button.inlineDownloadPDF`} inline />
                          ),
                        }
                      : {}
                  }
                />
              ),
            }}
          />
        </>
      ) : (
        <>
          {mefErrorCodes.length > 1 ? (
            !isAfterResubmissionDeadline && (
              <ContentDisplay
                i18nKey='taxReturnDetails.rejectedReturnDetails.fixable.multipleErrors.howToFix'
                collectionId={null}
              />
            )
          ) : (
            <>
              <h2>{t(`taxReturnDetails.rejectedReturnDetails.errorCode`, { errorCode: mefErrorCodes[0] })}</h2>
              {!isAfterResubmissionDeadline && (
                <ContentDisplay
                  i18nKey='taxReturnDetails.rejectedReturnDetails.fixable.howToFix'
                  context={{ count: mefErrorCodes.length, errorCode: mefErrorCodes[0] }}
                  collectionId={null}
                />
              )}
            </>
          )}
        </>
      )}
    </>
  );
};

const UnfixableReturnDetails = ({ mefErrorCodes, taxYear }: ReturnDetailsProps) => {
  const { t, i18n } = useTranslation(`translation`);
  const [isBeforeFederalTaxDeadline] = useFact<boolean>(Path.concretePath(`/isBeforeFederalTaxDeadline`, null));
  const singleErrorKeyPrefix = `taxReturnDetails.rejectedReturnDetails.unfixable.singleErrorExplanation`;
  const hasOneMefErrorWithContent =
    mefErrorCodes.length === 1 && i18n.exists(`${singleErrorKeyPrefix}.${mefErrorCodes[0]}`);

  const commonUrls = useCommonUrls(t);

  return (
    <>
      {hasOneMefErrorWithContent ? (
        <>
          <h2>{t(`taxReturnDetails.rejectedReturnDetails.errorCode`, { errorCode: mefErrorCodes[0] })}</h2>
          <ContentDisplay
            i18nKey={`${singleErrorKeyPrefix}.${mefErrorCodes[0]}.rejectionDetails`}
            context={{ taxYear: taxYear }}
            collectionId={null}
            additionalComponents={commonUrls}
          />
          <ContentDisplay i18nKey={`${singleErrorKeyPrefix}.${mefErrorCodes[0]}.newTool`} />
        </>
      ) : (
        <>
          <p>{t(`taxReturnDetails.rejectedReturnDetails.unfixable.couldNotProcessReturn`)}</p>
        </>
      )}
      <h2>{t(`taxReturnDetails.rejectedReturnDetails.unfixable.findAnotherWayToFile.heading`)}</h2>
      <ContentDisplay
        i18nKey='taxReturnDetails.rejectedReturnDetails.unfixable.findAnotherWayToFile.main'
        collectionId={null}
        context={{ count: mefErrorCodes.length }}
      />
      {isBeforeFederalTaxDeadline && (
        <ContentDisplay
          i18nKey='taxReturnDetails.rejectedReturnDetails.unfixable.findAnotherWayToFile.extra'
          collectionId={null}
          context={{ count: mefErrorCodes.length }}
        />
      )}
    </>
  );
};

const RejectedReturnDetails = ({ taxReturn, submissionStatus }: RejectedReturnDetailsProps) => {
  const { t, i18n } = useTranslation(`translation`);

  // TODO: Delete these facts or action https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/8045
  //       In the context of DF staying open through October, they might not be needed at all?
  //       The related logic below has become complex depending on these values
  //       (which are, at the time of writing, constant)
  const [isBeforeFederalTaxDeadline] = useFact<boolean>(Path.concretePath(`/isBeforeFederalTaxDeadline`, null));
  const [isAfterResubmissionDeadline] = useFact<boolean>(Path.concretePath(`/isAfterResubmissionDeadline`, null));

  const mefErrorCodes =
    submissionStatus.rejectionCodes.length > 0 ? submissionStatus.rejectionCodes.map((rc) => rc.MeFErrorCode) : [];
  const hasUnfixableMefErrors = areAnyRejectionsNotFixable(submissionStatus.rejectionCodes);
  const formattedFilingDate = formatAsContentDate(FILING_DEADLINE, i18n);

  return (
    <>
      <div className='screen__header'>
        <h1 className='screen-heading'>
          {hasUnfixableMefErrors &&
            t(`taxReturnDetails.rejectedReturnDetails.unfixable.heading`, {
              taxYear: taxReturn.taxYear,
            })}
          {!hasUnfixableMefErrors &&
            (isAfterResubmissionDeadline
              ? t(`taxReturnDetails.rejectedReturnDetails.fixable.postResubmissionDeadlineHeading`, {
                  taxYear: taxReturn.taxYear,
                  count: submissionStatus?.rejectionCodes.length,
                })
              : t(`taxReturnDetails.rejectedReturnDetails.fixable.heading`, {
                  taxYear: taxReturn.taxYear,
                  count: submissionStatus?.rejectionCodes.length,
                }))}
        </h1>
      </div>
      {isAfterResubmissionDeadline && (
        <ContentDisplay i18nKey='taxReturnDetails.rejectedReturnDetails.fixable.preamble' collectionId={null} />
      )}
      {mefErrorCodes.length > 0 && (
        <div className='usa-prose'>
          {hasUnfixableMefErrors ? (
            <UnfixableReturnDetails mefErrorCodes={mefErrorCodes} taxYear={taxReturn.taxYear} />
          ) : (
            <FixableReturnDetails mefErrorCodes={mefErrorCodes} taxYear={taxReturn.taxYear} />
          )}
          {mefErrorCodes.length > 1 && (
            <>
              <Accordion
                className='margin-top-2'
                items={mefErrorCodes.map((mefErrorCode, index) => {
                  return {
                    headingLevel: `h2`,
                    title: t(`taxReturnDetails.rejectedReturnDetails.fixable.multipleErrors.title`, {
                      number: index + 1,
                      errorCode: mefErrorCode,
                    }),
                    content: <FixableReturnDetails mefErrorCodes={[mefErrorCode]} taxYear={taxReturn.taxYear} />,
                    expanded: false,
                    id: `${index}_${mefErrorCode}`,
                  };
                })}
              />
            </>
          )}
          {!hasUnfixableMefErrors && !isAfterResubmissionDeadline && (
            <>
              <h2>{t(`taxReturnDetails.rejectedReturnDetails.fixable.customerSupportHeading`)}</h2>
              <p>
                <Translation
                  i18nKey='taxReturnDetails.rejectedReturnDetails.fixable.customerSupport'
                  collectionId={null}
                />
              </p>
              <h2>{t(`taxReturnDetails.rejectedReturnDetails.fixable.resubmitDeadline.heading`)}</h2>
              <ContentDisplay
                i18nKey='taxReturnDetails.rejectedReturnDetails.fixable.resubmitDeadline'
                collectionId={null}
                context={{ filingDeadline: formattedFilingDate }}
              />
            </>
          )}
        </div>
      )}
      <>
        {hasUnfixableMefErrors && (
          <>
            <CommonLinkRenderer className='usa-button width-full' url={t(`commonUrls.otherWaysToFile`)}>
              {t(`taxReturnDetails.rejectedReturnDetails.unfixable.findOtherWaysToFileButtonText`)}
            </CommonLinkRenderer>
            {isBeforeFederalTaxDeadline && (
              <IntroContent
                i18nKey='taxReturnDetails.rejectedReturnDetails.unfixable.findAnotherWayToFile.extension'
                collectionId={null}
              />
            )}
          </>
        )}
        {!hasUnfixableMefErrors && isAfterResubmissionDeadline ? (
          <>
            <CommonLinkRenderer className='usa-button width-full' url={t(`commonUrls.otherWaysToFile`)}>
              {t(`taxReturnDetails.rejectedReturnDetails.unfixable.findOtherWaysToFileButtonText`)}
            </CommonLinkRenderer>
            <CommonLinkRenderer className='usa-button usa-button--outline width-full' url='/checklist'>
              {t(`taxReturnDetails.rejectedReturnDetails.fixable.revisitReturnButtonText`, {
                taxYear: taxReturn.taxYear,
              })}
            </CommonLinkRenderer>
          </>
        ) : (
          !hasUnfixableMefErrors && (
            <CommonLinkRenderer className='usa-button width-full' url='/checklist'>
              {t(`taxReturnDetails.rejectedReturnDetails.fixable.editReturnButtonText`, { taxYear: taxReturn.taxYear })}
            </CommonLinkRenderer>
          )
        )}
      </>
    </>
  );
};

export default RejectedReturnDetails;
