import { useContext } from 'react';
import { Helmet } from 'react-helmet-async';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { CommonContentDisplay } from '@irs/df-common';
import { Alert } from '@trussworks/react-uswds';

import { Heading, Translation, Prose, NextLink, Breadcrumbs, SubHeader } from '../../components/index.js';
import { PilotPhaseContext } from '../../layouts/Providers.js';

const Identification = () => {
  const { t } = useTranslation(`translation`);
  const navigate = useNavigate();
  const phase = useContext(PilotPhaseContext);
  const PREV = `/state`;
  const NEXT = `/income`;

  if (!phase?.enableScreener) navigate(`/`);

  return (
    <>
      <Helmet>
        <title>{t(`pages.ScreenerIdentification.pageTitle`)}</title>
      </Helmet>
      <SubHeader />
      <Breadcrumbs href={PREV} />
      {/* <StepIndicator currentStepKey='ScreenerIdentification' /> */}
      <Prose>
        <Heading>{t(`pages.ScreenerIdentification.heading`)}</Heading>
        <CommonContentDisplay i18nKey='pages.ScreenerIdentification' TranslationComponent={Translation} />
        <Alert className='margin-top-3' type='warning' headingLevel='h3' validation>
          <CommonContentDisplay i18nKey='pages.ScreenerIdentification.alert' TranslationComponent={Translation} />
        </Alert>
      </Prose>
      <NextLink href={NEXT}>{t(`pages.ScreenerIdentification.button.text`)}</NextLink>
    </>
  );
};

export default Identification;
