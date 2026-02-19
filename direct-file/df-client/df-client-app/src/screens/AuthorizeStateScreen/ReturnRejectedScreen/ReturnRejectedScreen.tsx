import { Trans, useTranslation } from 'react-i18next';
import { Button, GridContainer } from '@trussworks/react-uswds';
import StackedButtonGroup from '../../../components/StackedButtonGroup/StackedButtonGroup.js';
import { FederalReturnStatus, TaxReturn } from '../../../types/core.js';
import { MouseEventHandler } from 'react';
import StatusInfo from '../StatusInfo/StatusInfo.js';
import { StateOrProvince } from '../../../types/StateOrProvince.js';
import PageTitle from '../../../components/PageTitle/index.js';
import useTranslatePIIRedacted from '../../../hooks/useTranslatePIIRedacted.js';

export type RejectedReturnScreenProps = {
  taxYear: TaxReturn[`taxYear`];
  taxReturnStatus: FederalReturnStatus;
  stateCode: StateOrProvince;
  stateTaxSystemName: string;
  federalReturnMustBeAccepted: boolean;
  handleGoBack: MouseEventHandler<HTMLButtonElement>;
};

const RejectedReturnScreen = ({
  taxYear,
  taxReturnStatus,
  stateCode,
  stateTaxSystemName,
  federalReturnMustBeAccepted,
  handleGoBack,
}: RejectedReturnScreenProps) => {
  const { t } = useTranslation(`translation`, { keyPrefix: `authorizeState.rejectedReturn` });
  const { t: tStates } = useTranslation(`translation`, { keyPrefix: `enums.statesAndProvinces` });
  const redacted = useTranslatePIIRedacted(`authorizeState.rejectedReturn.heading`, true, {
    taxYear: taxYear.toString(),
  });

  return (
    <GridContainer>
      <PageTitle redactedTitle={redacted}>{t(`heading`, { taxYear: taxYear.toString() })}</PageTitle>
      <div className='usa-prose margin-top-3'>
        <StatusInfo taxReturnStatus={taxReturnStatus}>
          {federalReturnMustBeAccepted ? (
            <Trans
              t={t}
              i18nKey={`statusInfoAcceptedOnly`}
              values={{ stateName: tStates(stateCode), stateTaxSystemName }}
            />
          ) : (
            <Trans
              t={t}
              i18nKey={`statusInfoMayBePending`}
              values={{ stateName: tStates(stateCode), stateTaxSystemName }}
            />
          )}
        </StatusInfo>
      </div>
      <StackedButtonGroup>
        <Button type='button' unstyled onClick={handleGoBack}>
          {t(`goBack`)}
        </Button>
      </StackedButtonGroup>
    </GridContainer>
  );
};

export default RejectedReturnScreen;
