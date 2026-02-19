import { Helmet } from 'react-helmet-async';
import { useTranslation } from 'react-i18next';

import { CommonContentDisplay } from '@irs/df-common';

import { Heading, Translation, Prose, Breadcrumbs } from '../components/index.js';

const PilotLimitedAccess = () => {
  const { t } = useTranslation(`translation`);
  const PREV = `/done`;

  return (
    <>
      <Helmet>
        <title>{t(`pages.PilotLimitedAccess.pageTitle`)}</title>
      </Helmet>
      <Breadcrumbs href={PREV} />
      <Prose>
        <Heading level='h1' large>
          {t(`pages.PilotLimitedAccess.heading`)}
        </Heading>
        <CommonContentDisplay
          i18nKey='pages.PilotLimitedAccess'
          allowedTags={[`p`, `h2`]}
          TranslationComponent={Translation}
        />
      </Prose>
    </>
  );
};

export default PilotLimitedAccess;
