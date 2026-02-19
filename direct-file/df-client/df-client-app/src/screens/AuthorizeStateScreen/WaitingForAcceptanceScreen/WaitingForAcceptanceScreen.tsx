import { Alert, Button, GridContainer, Link, ModalRef } from '@trussworks/react-uswds';
import StackedButtonGroup from '../../../components/StackedButtonGroup/StackedButtonGroup.js';
import { Trans, useTranslation } from 'react-i18next';
import { FederalReturnStatus, TaxReturn } from '../../../types/core.js';
import StatusInfo from '../StatusInfo/StatusInfo.js';
import { MouseEventHandler, useContext, useEffect, useRef, useState } from 'react';
import StatusInfoModal from '../StatusInfoModal/StatusInfoModal.js';
import { StateOrProvince } from '../../../types/StateOrProvince.js';
import PageTitle from '../../../components/PageTitle/index.js';
import useTranslatePIIRedacted from '../../../hooks/useTranslatePIIRedacted.js';
import styles from './WaitingForAcceptanceScreen.module.scss';
import { formatAsDateTimeString } from '../../../utils/dateUtils.js';
import { SubmissionStatusContext } from '../../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import SystemAlertAggregator from '../../../components/SystemAlertAggregator/SystemAlertAggregator.js';

export const STATUS_FETCH_LOCKOUT_SECONDS = 30;

export type WaitingForAcceptanceScreenProps = {
  taxYear: TaxReturn[`taxYear`];
  taxReturnStatus: FederalReturnStatus;
  stateCode: StateOrProvince;
  stateTaxSystemName: string;
  goBackUrl: URL;
  taxReturnId: string;
};

const WaitingForAcceptanceScreen = ({
  taxYear,
  taxReturnStatus,
  stateCode,
  stateTaxSystemName,
  goBackUrl,
  taxReturnId,
}: WaitingForAcceptanceScreenProps) => {
  const { t, i18n } = useTranslation(`translation`, { keyPrefix: `authorizeState.waitingForAcceptance` });
  const { t: tStatusUpdateModal } = useTranslation(`translation`, { keyPrefix: `authorizeState.statusUpdateModal` });
  const { t: tStates } = useTranslation(`translation`, { keyPrefix: `enums.statesAndProvinces` });

  const { fetchSubmissionStatus, lastFetchAttempt: lastStatusFetchAttempt } = useContext(SubmissionStatusContext);

  const modalRef = useRef<ModalRef>(null);

  const getSecondsSinceLastFetch = () =>
    lastStatusFetchAttempt ? Math.round((new Date().getTime() - lastStatusFetchAttempt.getTime()) / 1000) : 0;

  const [statusFetchLockoutSecondsRemaining, setStatusFetchLockoutSecondsRemaining] = useState(
    STATUS_FETCH_LOCKOUT_SECONDS - getSecondsSinceLastFetch()
  );
  const [showStatusFetchLockoutMessage, setShowStatusFetchLockoutMessage] = useState(false);

  // Count down timer management
  useEffect(() => {
    if (statusFetchLockoutSecondsRemaining > 0) {
      // Count down until we hit 0
      const interval = setInterval(() => {
        setStatusFetchLockoutSecondsRemaining((currentLockoutSeconds) => currentLockoutSeconds - 1);
      }, 1000);
      return () => clearInterval(interval);
    } else {
      // once at (or below) 0
      if (showStatusFetchLockoutMessage) {
        // if the user has triggered the lockout message, fetch status again
        setShowStatusFetchLockoutMessage(false);
        fetchSubmissionStatus(taxReturnId);
        // At runtime, fetching status effectively reloads this entire component, therefore setting state to the
        // default value as this component is freshly mounted. However, for all-screens:
        setStatusFetchLockoutSecondsRemaining(STATUS_FETCH_LOCKOUT_SECONDS);
      }
    }
  }, [
    statusFetchLockoutSecondsRemaining,
    setStatusFetchLockoutSecondsRemaining,
    fetchSubmissionStatus,
    taxReturnId,
    showStatusFetchLockoutMessage,
  ]);

  const handleClickStatusInfoButton: MouseEventHandler<HTMLButtonElement> = (e) => {
    modalRef.current?.toggleModal(e, true);
  };

  const handleClickCheckStatusButton: MouseEventHandler<HTMLButtonElement> = (e) => {
    const secondsSinceLastFetch = getSecondsSinceLastFetch();
    const canCheckStatus = secondsSinceLastFetch >= STATUS_FETCH_LOCKOUT_SECONDS;
    if (canCheckStatus) {
      fetchSubmissionStatus(taxReturnId);
      // At runtime, fetching status effectively reloads this entire component, therefore setting state to the
      // default value as this component is freshly mounted. However, for all-screens:
      setStatusFetchLockoutSecondsRemaining(STATUS_FETCH_LOCKOUT_SECONDS);
    } else {
      setShowStatusFetchLockoutMessage(true);
      e.currentTarget.blur();
    }
  };

  const redacted = useTranslatePIIRedacted(`authorizeState.waitingForAcceptance.heading`, true, {
    taxYear: taxYear.toString(),
  });

  return (
    <>
      <StatusInfoModal modalRef={modalRef} canTransfer={false} />
      <GridContainer>
        <SystemAlertAggregator />
        <PageTitle redactedTitle={redacted}>{t(`heading`, { taxYear: taxYear.toString() })}</PageTitle>
        <div className='usa-prose margin-top-3'>
          <StatusInfo taxReturnStatus={taxReturnStatus}>
            <Trans t={t} i18nKey={`statusInfo`} values={{ stateName: tStates(stateCode), stateTaxSystemName }} />
            <Button type='button' unstyled className='display-block margin-top-3' onClick={handleClickStatusInfoButton}>
              {tStatusUpdateModal(`modalOpenButtonText`)}
            </Button>
            {lastStatusFetchAttempt && (
              <>
                <br />
                <span className={styles.lastStatusCheck}>
                  <Trans
                    t={t}
                    i18nKey={`lastStatusCheck`}
                    values={{ lastStatusCheck: formatAsDateTimeString(i18n.language, lastStatusFetchAttempt) }}
                  />
                </span>
              </>
            )}
          </StatusInfo>
        </div>
        {showStatusFetchLockoutMessage && (
          <Alert type='warning' headingLevel='h2' data-testid='status-fetch-lockout-alert'>
            {t(`checkingStatus`, {
              count: statusFetchLockoutSecondsRemaining,
            })}
          </Alert>
        )}
        <StackedButtonGroup>
          {!showStatusFetchLockoutMessage && (
            <Button type='button' className='margin-top-2' outline onClick={handleClickCheckStatusButton}>
              {t(`checkStatusButtonText`)}
            </Button>
          )}

          <Link variant='external' href={goBackUrl.toString()}>
            {t(`goBack`, { stateTaxSystemName })}
          </Link>
        </StackedButtonGroup>
      </GridContainer>
    </>
  );
};

export default WaitingForAcceptanceScreen;
