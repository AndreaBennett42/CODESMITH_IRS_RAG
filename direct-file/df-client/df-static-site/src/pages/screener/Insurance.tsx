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
  DFIconList,
  DFIconListItem,
  SubHeader,
} from '../../components/index.js';
import { PilotPhaseContext } from '../../layouts/Providers.js';

const Insurance = () => {
  const { t } = useTranslation(`translation`);
  const navigate = useNavigate();
  const phase = useContext(PilotPhaseContext);
  const PREV = `/deductions`;
  const NEXT = `/credits`;

  if (!phase?.enableScreener) navigate(`/`);

  const allowedInsuranceList = `pages.ScreenerInsurance.allowedInsuranceList`;
  const disallowedList = `pages.ScreenerInsurance.disallowedList`;

  return (
    <>
      <Helmet>
        <title>{t(`pages.ScreenerInsurance.pageTitle`)}</title>
      </Helmet>
      <SubHeader />
      <Breadcrumbs href={PREV} />
      {/* <StepIndicator currentStepKey='ScreenerInsurance' /> */}
      <Prose>
        <Heading>{t(`pages.ScreenerInsurance.heading`)}</Heading>
        <CommonContentDisplay
          allowedTags={[`h3`]}
          i18nKey='pages.ScreenerInsurance'
          TranslationComponent={Translation}
        />
        <DFIconList>
          <DFIconListItem mood='good' i18nKey={`${allowedInsuranceList}.no`} />
          <DFIconListItem mood='good' i18nKey={`${allowedInsuranceList}.employer`} />
          <DFIconListItem mood='good' i18nKey={`${allowedInsuranceList}.medicare`} />
          <DFIconListItem mood='good' i18nKey={`${allowedInsuranceList}.va`} />
          <DFIconListItem mood='good' i18nKey={`${allowedInsuranceList}.private`} />
          <DFIconListItem mood='good' i18nKey={`${allowedInsuranceList}.medicaid`} hasHTML />
        </DFIconList>
        <CommonContentDisplay
          allowedTags={[`h3`]}
          i18nKey='pages.ScreenerInsurance.cannot-explainer'
          TranslationComponent={Translation}
        />
        <DFIconList>
          <DFIconListItem mood='bad' i18nKey={`${disallowedList}.health-savings`} />
          <DFIconListItem mood='bad' i18nKey={`${disallowedList}.marketplace`} hasHTML />
        </DFIconList>
        <DFAccordion i18nKey='pages.ScreenerInsurance.MarketPlace' />
        <CommonContentDisplay i18nKey='pages.ScreenerInsurance.outro' TranslationComponent={Translation} />
      </Prose>
      <NextLink href={NEXT}>{t(`pages.ScreenerInsurance.button.text`)}</NextLink>
    </>
  );
};

export default Insurance;
