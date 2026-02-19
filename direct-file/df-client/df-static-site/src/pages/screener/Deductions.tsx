import { useContext } from 'react';
import { Helmet } from 'react-helmet-async';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { CommonContentDisplay } from '@irs/df-common';

import {
  Heading,
  Translation,
  DFAccordion,
  Prose,
  NextLink,
  Breadcrumbs,
  StepIndicator,
  DFIconList,
  DFIconListItem,
  SubHeader,
} from '../../components/index.js';
import { PilotPhaseContext } from '../../layouts/Providers.js';

const Deductions = () => {
  const { t } = useTranslation(`translation`);
  const navigate = useNavigate();
  const phase = useContext(PilotPhaseContext);
  const PREV = `/savingsandretirement`;
  const NEXT = `/credits`;

  if (!phase?.enableScreener) navigate(`/`);

  const deductionTypeList = `pages.ScreenerDeductions.deductionTypeList`;

  return (
    <>
      <Helmet>
        <title>{t(`pages.ScreenerDeductions.pageTitle`)}</title>
      </Helmet>
      <SubHeader />
      <Breadcrumbs href={PREV} />
      <StepIndicator currentStepKey='ScreenerDeductions' />
      <Prose>
        <Heading>{t(`pages.ScreenerDeductions.heading`)}</Heading>
        <CommonContentDisplay i18nKey='pages.ScreenerDeductions' TranslationComponent={Translation} />
        <DFIconList>
          <DFIconListItem mood='good' i18nKey={`${deductionTypeList}.standard`} />
          <DFIconListItem mood='bad' i18nKey={`${deductionTypeList}.itemized`} />
        </DFIconList>
        <CommonContentDisplay
          i18nKey='pages.ScreenerDeductions.popularity-explainer'
          TranslationComponent={Translation}
        />
        <DFAccordion i18nKey='pages.ScreenerDeductions.StandardDeduction' />
        <DFAccordion i18nKey='pages.ScreenerDeductions.ItemizedDeductions' />
        <CommonContentDisplay i18nKey='pages.ScreenerDeductions.outro' TranslationComponent={Translation} />
      </Prose>
      <NextLink href={NEXT}>{t(`pages.ScreenerDeductions.button.text`)}</NextLink>
    </>
  );
};

export default Deductions;
