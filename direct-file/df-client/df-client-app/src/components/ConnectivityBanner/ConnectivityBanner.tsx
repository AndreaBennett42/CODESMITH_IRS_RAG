import { Alert } from '@trussworks/react-uswds';
import { Trans, useTranslation } from 'react-i18next';
import { useContext } from 'react';
import { NetworkConnectionContext } from '../../context/networkConnectionContext.js';

const ConnectivityBanner = () => {
  const { t } = useTranslation(`translation`);
  const { online } = useContext(NetworkConnectionContext);
  const networkStatus = online ? `online` : `offline`;

  return (
    // TODO: replace this with a generic Banner component which wraps the Alert component -RL
    <section aria-label={t(`banner.connectivity.name`)}>
      <Alert type={online ? `info` : `warning`} headingLevel='h6' noIcon role='alert'>
        <Trans>{t(`banner.connectivity.${networkStatus}.not_accessible`)}</Trans>
      </Alert>
    </section>
  );
};
export default ConnectivityBanner;
