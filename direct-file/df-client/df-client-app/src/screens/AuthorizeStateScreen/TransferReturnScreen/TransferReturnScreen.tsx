import { MouseEventHandler, useContext, useRef } from 'react';
import { FederalReturnStatus, TaxReturn } from '../../../types/core.js';
import { Trans, useTranslation } from 'react-i18next';
import { Button, GridContainer, Link, ModalRef } from '@trussworks/react-uswds';
import { Link as RouterLink } from 'react-router-dom';
import StackedButtonGroup from '../../../components/StackedButtonGroup/StackedButtonGroup.js';
import StatusInfo from '../StatusInfo/StatusInfo.js';
import { FEDERAL_RETURN_STATUS } from '../../../constants/taxConstants.js';
import StatusInfoModal from '../StatusInfoModal/StatusInfoModal.js';
import TransferInfoModal from '../TransferInfoModal/TransferInfoModal.js';
import { StateOrProvince } from '../../../types/StateOrProvince.js';
import PageTitle from '../../../components/PageTitle/index.js';
import useTranslatePIIRedacted from '../../../hooks/useTranslatePIIRedacted.js';
import styles from './TransferReturnScreen.module.scss';
import { formatAsDateTimeString } from '../../../utils/dateUtils.js';
import { SubmissionStatusContext } from '../../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import SystemAlertAggregator from '../../../components/SystemAlertAggregator/SystemAlertAggregator.js';

export type TransferReturnScreenProps = {
  taxYear: TaxReturn[`taxYear`];
  taxReturnStatus: FederalReturnStatus;
  taxReturnUuid: string;
  stateCode: StateOrProvince;
  stateTaxSystemName: string;
  handleSubmit: MouseEventHandler<HTMLButtonElement>;
  goBackUrl: URL;
  isSubmittingTransfer: boolean;
};
const TransferReturnScreen = ({
  taxYear,
  taxReturnStatus,
  taxReturnUuid,
  stateCode,
  stateTaxSystemName,
  handleSubmit,
  goBackUrl,
  isSubmittingTransfer,
}: TransferReturnScreenProps) => {
  const { t, i18n } = useTranslation(`translation`, { keyPrefix: `authorizeState.transferReturn` });
  const { t: tStatusInfoModal } = useTranslation(`translation`, { keyPrefix: `authorizeState.statusInfoModal` });
  const { t: tStates } = useTranslation(`translation`, { keyPrefix: `enums.statesAndProvinces` });

  const { lastFetchAttempt: lastStatusFetchAttempt } = useContext(SubmissionStatusContext);

  const transferInfoModalRef = useRef<ModalRef>(null);
  const statusInfoModalRef = useRef<ModalRef>(null);

  const stateName = tStates(stateCode);

  const handleClickStatusInfoButton: MouseEventHandler<HTMLButtonElement> = (e) => {
    statusInfoModalRef.current?.toggleModal(e, true);
  };

  const handleClickTransferInfoButton: MouseEventHandler<HTMLButtonElement> = (e) => {
    transferInfoModalRef.current?.toggleModal(e, true);
  };
  const redactedContext = { taxYear: taxYear.toString(), stateName, stateTaxSystemName };
  const redacted = useTranslatePIIRedacted(`authorizeState.transferReturn.heading`, true, redactedContext);

  return (
    <>
      {taxReturnStatus === FEDERAL_RETURN_STATUS.PENDING && (
        <StatusInfoModal modalRef={statusInfoModalRef} canTransfer />
      )}
      <TransferInfoModal modalRef={transferInfoModalRef} taxReturnUuid={taxReturnUuid} />
      <GridContainer>
        <SystemAlertAggregator />
        <PageTitle redactedTitle={redacted}>{t(`heading`, redactedContext)}</PageTitle>
        <div className='usa-prose margin-top-3'>
          <StatusInfo taxReturnStatus={taxReturnStatus}>
            {taxReturnStatus === FEDERAL_RETURN_STATUS.PENDING && (
              <>
                <Trans t={t} i18nKey={`statusInfo.pending`} />
                <br />
                <br />
                <Button type='button' unstyled onClick={handleClickStatusInfoButton}>
                  {tStatusInfoModal(`modalOpenButtonText`)}
                </Button>
                <br />
                <br />
                {lastStatusFetchAttempt && (
                  <span className={styles.lastStatusCheck}>
                    <Trans
                      t={t}
                      i18nKey={`statusInfo.lastStatusCheck`}
                      values={{ lastStatusCheck: formatAsDateTimeString(i18n.language, lastStatusFetchAttempt) }}
                    />
                  </span>
                )}
              </>
            )}
            {taxReturnStatus === FEDERAL_RETURN_STATUS.REJECTED && (
              <Trans t={t} i18nKey={`statusInfo.rejected`}>
                <RouterLink to='/home'>Direct File dashboard</RouterLink>
              </Trans>
            )}
          </StatusInfo>
          <p>{t(`attestation`, { stateName, stateTaxSystemName })}</p>
          <Button unstyled type='button' className='margin-top-2' onClick={handleClickTransferInfoButton}>
            {t(`transferInfoButtonText`)}
          </Button>
        </div>
        <StackedButtonGroup>
          <Button type='submit' size='big' onClick={handleSubmit} disabled={isSubmittingTransfer}>
            <p className='usa-link--external margin-0'>
              {t(`submitButtonText`, { taxYear: taxYear.toString(), stateTaxSystemName })}
            </p>
          </Button>
          <Link variant='external' href={goBackUrl.toString()}>
            {t(`cancel`, { stateTaxSystemName })}
          </Link>
        </StackedButtonGroup>
      </GridContainer>
    </>
  );
};

export default TransferReturnScreen;
