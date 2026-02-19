import { FC, useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useLocation } from 'react-router-dom';
import { Alert } from '@trussworks/react-uswds';
import { CommonContentDisplay } from '@irs/df-common';
import { PilotPhaseContext } from '../../layouts/Providers.js';
import { Translation } from '../index.js';
import styles from './PilotBanner.module.scss';

const PilotBanner: FC = () => {
  const { t } = useTranslation(`translation`);
  const phase = useContext(PilotPhaseContext);
  const { pathname } = useLocation();
  if (phase?.showOpenAccessBanner) {
    return;
  } else if (phase?.showClosingSoonBanner) {
    return (
      <Alert type='warning' headingLevel='h1' className={`${styles.dfPilotBanner} margin-top-0`} validation>
        <h1 className='font-sans-sm'>{t(`components.PilotBanner.closing-soon.header`)}</h1>
        {pathname === `/` && (
          <CommonContentDisplay
            i18nKey='components.PilotBanner.closing-soon'
            allowedTags={[`p`, `h3`]}
            TranslationComponent={Translation}
          />
        )}
      </Alert>
    );
  } else if (phase?.showAfterDeadlineBanner) {
    return (
      <Alert type='warning' headingLevel='h1' className={`${styles.dfPilotBanner} margin-top-0`} validation>
        <h1 className='font-sans-sm'>{t(`components.PilotBanner.after-deadline.header`)}</h1>
        {pathname === `/` && (
          <CommonContentDisplay
            i18nKey='components.PilotBanner.after-deadline'
            allowedTags={[`p`, `h3`]}
            TranslationComponent={Translation}
          />
        )}
      </Alert>
    );
  } else if (phase?.showPilotClosedBanner) {
    return (
      <Alert type='warning' headingLevel='h1' className={`${styles.dfPilotBanner} margin-top-0`} validation>
        <h1 className='font-sans-sm'>{t(`components.PilotBanner.closed.header`)}</h1>
        <CommonContentDisplay
          i18nKey='components.PilotBanner.closed'
          allowedTags={[`p`, `h3`]}
          TranslationComponent={Translation}
        />
      </Alert>
    );
  } else {
    return null;
  }
};

export default PilotBanner;
