import { useTranslation } from 'react-i18next';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import ContentDisplay from '../../../components/ContentDisplay/index.js';
import { TaxReturn } from '../../../types/core.js';
import useFact from '../../../hooks/useFact.js';
import { Path } from '../../../flow/Path.js';
import { formatAsContentDate } from '../../../utils/dateUtils.js';
import { FILING_DEADLINE } from '../../../constants/taxConstants.js';

type ErroredReturnDetailsProps = {
  taxReturn: TaxReturn;
};

// Note: This component is a stopgap for a planned P2 and has not gone through standard design + testing / review
const ErroredReturnDetails = ({ taxReturn }: ErroredReturnDetailsProps) => {
  const { t, i18n } = useTranslation(`translation`);
  const [isBeforeFederalTaxDeadline] = useFact<boolean>(Path.concretePath(`/isBeforeFederalTaxDeadline`, null));
  const { taxYear } = taxReturn;
  const erroredReturnI18nKey = isBeforeFederalTaxDeadline ? `erroredReturnDetails` : `erroredReturnDetailsPostDeadline`;
  const formattedFilingDate = formatAsContentDate(FILING_DEADLINE, i18n);

  return (
    <>
      <div className='screen__header'>
        <h1 className='screen-heading'>{t(`taxReturnDetails.erroredReturnDetails.heading`, { taxYear })}</h1>
      </div>
      <h2>{t(`taxReturnDetails.erroredReturnDetails.whatToDoNext`)}</h2>
      <ContentDisplay
        i18nKey={`taxReturnDetails.${erroredReturnI18nKey}`}
        collectionId={null}
        context={{ taxYear, filingDeadline: formattedFilingDate }}
      />
      <div className='screen__actions'>
        <CommonLinkRenderer className='usa-button width-full' url={t(`commonUrls.otherWaysToFile`)}>
          {t(`taxReturnCard.otherWaysToFile`)}
        </CommonLinkRenderer>
        <CommonLinkRenderer className='usa-button usa-button--outline width-full' url='/checklist'>
          {t(`taxReturnDetails.erroredReturnDetails.revisitReturnButtonText`, { taxYear })}
        </CommonLinkRenderer>
      </div>
    </>
  );
};

export default ErroredReturnDetails;
