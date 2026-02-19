import { Alert } from '@trussworks/react-uswds';
import { useTranslation } from 'react-i18next';
import { BareContentDisplay } from '../ContentDisplay/ContentDisplay.js';

/** A tiny component to give users a warning that the tax deadline is approaching. */
const ClosingSoonBanner = () => {
  const { t } = useTranslation(`translation`);

  return (
    <section aria-label={t(`banner.closingSoon.name`)}>
      <Alert type='warning' headingLevel='h1' className='margin-top-0' role='alert' validation>
        <h1 className='font-sans-sm'>{t(`banner.closingSoon.header`)}</h1>
        <BareContentDisplay i18nKey='banner.closingSoon' allowedTags={[`p`]} />
      </Alert>
    </section>
  );
};
export default ClosingSoonBanner;
