import { useContext } from 'react';
import { Helmet } from 'react-helmet-async';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { CommonContentDisplay } from '@irs/df-common';
import { Alert } from '@trussworks/react-uswds';

import {
  Heading,
  Translation,
  Prose,
  DFIconListItem,
  DFIconList,
  NextLink,
  Breadcrumbs,
  StepIndicator,
  DFAccordion,
  SubHeader,
} from '../../components/index.js';
import { PilotPhaseContext } from '../../layouts/Providers.js';

const Income = () => {
  const { t } = useTranslation(`translation`);
  const navigate = useNavigate();
  const phase = useContext(PilotPhaseContext);
  const PREV = `/state`;
  const NEXT = `/savingsandretirement`;

  if (!phase?.enableScreener) navigate(`/`);

  const allowedIncomeList = `pages.ScreenerIncome.allowedIncomeList`;

  return (
    <>
      <Helmet>
        <title>{t(`pages.ScreenerIncome.pageTitle`)}</title>
      </Helmet>
      <SubHeader />
      <Breadcrumbs href={PREV} />
      <StepIndicator currentStepKey='ScreenerIncome' />
      <Prose>
        <Heading>{t(`pages.ScreenerIncome.heading`)}</Heading>
        <CommonContentDisplay i18nKey='pages.ScreenerIncome' TranslationComponent={Translation} />
        <DFIconList>
          <DFIconListItem mood='good' i18nKey={`${allowedIncomeList}.employer`} />
          <DFIconListItem mood='good' i18nKey={`${allowedIncomeList}.unemployment`} />
          <DFIconListItem mood='good' i18nKey={`${allowedIncomeList}.soc-sec`} />
          <DFIconListItem mood='good' i18nKey={`${allowedIncomeList}.retirement`} />
          <DFIconListItem mood='good' i18nKey={`${allowedIncomeList}.hsa`} />
          <DFIconListItem mood='good' i18nKey={`${allowedIncomeList}.interest`} />
          <DFIconListItem mood='good' i18nKey={`${allowedIncomeList}.alaska`} />
        </DFIconList>
        <Alert className='margin-top-3' type='warning' headingLevel='h3' validation>
          <CommonContentDisplay i18nKey='pages.ScreenerIncome.allowed-list-outro' TranslationComponent={Translation} />
        </Alert>
        <DFAccordion i18nKey='pages.ScreenerIncome.AdditionalLimitations' />
      </Prose>
      <NextLink href={NEXT}>{t(`pages.ScreenerIncome.button.text`)}</NextLink>
    </>
  );
};

export default Income;
