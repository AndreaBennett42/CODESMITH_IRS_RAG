import { useContext } from 'react';
import { Helmet } from 'react-helmet-async';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { CommonContentDisplay } from '@irs/df-common';

import {
  Heading,
  Translation,
  Prose,
  NextLink,
  Breadcrumbs,
  StepIndicator,
  DFIconListItem,
  DFAccordion,
  DFIconList,
  SubHeader,
} from '../../components/index.js';
import { PilotPhaseContext } from '../../layouts/Providers.js';

const Credits = () => {
  const { t } = useTranslation(`translation`);
  const navigate = useNavigate();
  const phase = useContext(PilotPhaseContext);
  const PREV = `/deductions`;
  const NEXT = `/done`;

  if (!phase?.enableScreener) navigate(`/`);

  const taxCreditList = `pages.ScreenerCredits.taxCreditList`;
  const disallowedList = `pages.ScreenerCredits.disallowedTaxCredits`;

  return (
    <>
      <Helmet>
        <title>{t(`pages.ScreenerCredits.pageTitle`)}</title>
      </Helmet>
      <SubHeader />
      <Breadcrumbs href={PREV} />
      <StepIndicator currentStepKey='ScreenerCredits' />
      <Prose>
        <Heading>{t(`pages.ScreenerCredits.heading`)}</Heading>
        <CommonContentDisplay i18nKey='pages.ScreenerCredits' TranslationComponent={Translation} />
        <DFIconList>
          <DFIconListItem mood='good' i18nKey={`${taxCreditList}.child-tax-credit`} hasHTML />
          <DFIconListItem mood='good' i18nKey={`${taxCreditList}.credit-other-dependents`} hasHTML />
          <DFIconListItem mood='good' i18nKey={`${taxCreditList}.credit-child-and-dependent`} hasHTML />
          <DFIconListItem mood='good' i18nKey={`${taxCreditList}.earned-income-tax-credit`} hasHTML />
          <DFIconListItem mood='good' i18nKey={`${taxCreditList}.premium-tax-credit`} hasHTML />
          <DFIconListItem mood='good' i18nKey={`${taxCreditList}.savers-credit`} hasHTML />
          <DFIconListItem mood='good' i18nKey={`${taxCreditList}.credit-elderly-or-disabled`} hasHTML />
        </DFIconList>
      </Prose>
      <Prose className='margin-top-3'>
        <CommonContentDisplay i18nKey='pages.ScreenerCredits.cannot-explainer' TranslationComponent={Translation} />
        <DFIconList>
          <DFIconListItem mood='bad' i18nKey={`${disallowedList}.other-credits`} />
          <DFIconListItem mood='bad' i18nKey={`${disallowedList}.non-custodial`} />
        </DFIconList>
      </Prose>
      <Prose className='margin-top-3'>
        <CommonContentDisplay i18nKey='pages.ScreenerCredits.examples' TranslationComponent={Translation} />
        <DFAccordion i18nKey='pages.ScreenerCredits.accordion' />
        <CommonContentDisplay i18nKey='pages.ScreenerCredits.outro' TranslationComponent={Translation} />
      </Prose>
      <NextLink href={NEXT}>{t(`pages.ScreenerCredits.button.text`)}</NextLink>
    </>
  );
};

export default Credits;
