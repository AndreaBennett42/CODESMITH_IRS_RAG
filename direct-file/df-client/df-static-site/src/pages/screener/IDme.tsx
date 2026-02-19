import { useContext } from 'react';
import { Helmet } from 'react-helmet-async';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { CommonContentDisplay } from '@irs/df-common';
import { Alert } from '@trussworks/react-uswds';

import { Heading, Translation, Prose, DFAccordion, NextLink, Breadcrumbs, SubHeader } from '../../components/index.js';
import { PilotPhaseContext } from '../../layouts/Providers.js';

const IDme = () => {
  const { t } = useTranslation(`translation`);
  const navigate = useNavigate();
  const phase = useContext(PilotPhaseContext);
  const PREV = `/credits`;
  const NEXT = `/done`;

  if (!phase?.enableScreener) navigate(`/`);

  return (
    <>
      <Helmet>
        <title>{t(`pages.ScreenerIDme.pageTitle`)}</title>
      </Helmet>
      <SubHeader />
      <Breadcrumbs href={PREV} />
      {/* <StepIndicator currentStepKey='ScreenerIDme' /> */}
      <Prose>
        <Heading>{t(`pages.ScreenerIDme.heading`)}</Heading>
        <CommonContentDisplay i18nKey='pages.ScreenerIDme' TranslationComponent={Translation} />
        <DFAccordion i18nKey='pages.ScreenerIDme.accordionText' />
        <Alert className='margin-top-3' type='warning' headingLevel='h3' validation>
          <CommonContentDisplay i18nKey='pages.ScreenerIDme.alert' TranslationComponent={Translation} />
        </Alert>
      </Prose>
      <NextLink href={NEXT}>{t(`pages.ScreenerIDme.button.text`)}</NextLink>
    </>
  );
};

export default IDme;
