import { Alert } from '@trussworks/react-uswds';

import styles from './StatusInfo.module.scss';
import { ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { FederalReturnStatus } from '../../../types/core.js';
import { FEDERAL_RETURN_STATUS } from '../../../constants/taxConstants.js';

export type StatusInfoProps = {
  taxReturnStatus: FederalReturnStatus;
  children?: ReactNode;
};

const getAlertTypeFrom = (status: FederalReturnStatus) => {
  switch (status) {
    case FEDERAL_RETURN_STATUS.ACCEPTED:
      return `success`;
    case FEDERAL_RETURN_STATUS.REJECTED:
      return `error`;
    default:
      return `info`;
  }
};

const StatusInfo = ({ taxReturnStatus, children }: StatusInfoProps) => {
  const { t } = useTranslation(`translation`, { keyPrefix: `authorizeState.statusInfo` });
  const { t: tStatus } = useTranslation(`translation`, { keyPrefix: `enums.status` });

  return (
    <Alert
      type={getAlertTypeFrom(taxReturnStatus)}
      headingLevel='h2'
      role='alert'
      heading={
        <span className={styles.headingText}>
          {t(`heading`)}: <span className={styles.headingStatusText}>{tStatus(taxReturnStatus)}</span>
        </span>
      }
      data-testid='status-info-alert'
    >
      {children}
    </Alert>
  );
};

export default StatusInfo;
