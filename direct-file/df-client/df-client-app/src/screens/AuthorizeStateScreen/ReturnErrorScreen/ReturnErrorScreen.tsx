import { GridContainer } from '@trussworks/react-uswds';
import { Trans, useTranslation } from 'react-i18next';
import { TaxReturn } from '../../../types/core.js';
import { FILING_DEADLINE } from '../../../constants/taxConstants.js';
import { formatAsContentDate } from '../../../utils/dateUtils.js';
import PageTitle from '../../../components/PageTitle/index.js';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import useTranslatePIIRedacted from '../../../hooks/useTranslatePIIRedacted.js';

export type ReturnErrorScreenProps = {
  taxYear: TaxReturn['taxYear'];
};

const ReturnErrorScreen = ({ taxYear }: ReturnErrorScreenProps) => {
  const { t: tRoot } = useTranslation(`translation`);
  const { t, i18n } = useTranslation(`translation`, { keyPrefix: `authorizeState.erroredReturn` });
  const redacted = useTranslatePIIRedacted(`authorizeState.erroredReturn.heading`, true, {
    taxYear: taxYear.toString(),
  });

  return (
    <GridContainer>
      <PageTitle redactedTitle={redacted}>{t(`heading`, { taxYear: taxYear.toString() })}</PageTitle>
      <div className='usa-prose margin-top-3'>
        <Trans
          t={t}
          i18nKey='content'
          values={{
            filingDeadline: formatAsContentDate(FILING_DEADLINE, i18n),
          }}
          components={{
            extensionApplicationLink: (
              <CommonLinkRenderer url={tRoot(`commonUrls.extensionApplication`)}>
                extensionApplicationLink
              </CommonLinkRenderer>
            ),
          }}
        >
          content
        </Trans>
      </div>
    </GridContainer>
  );
};

export default ReturnErrorScreen;
