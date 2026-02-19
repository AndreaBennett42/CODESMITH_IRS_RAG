import { useContext } from 'react';
import { Helmet } from 'react-helmet-async';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import {
  StepIndicator,
  SubHeader,
  Breadcrumbs,
  NextLink,
  Prose,
  Heading,
  Translation,
  DFIconList,
  DFIconListItem,
  DFAccordion,
} from '../../components/index.js';
import { CommonContentDisplay } from '@irs/df-common';
import { PilotPhaseContext } from '../../layouts/Providers.js';

const Retirement = () => {
  const { t } = useTranslation(`translation`);
  const navigate = useNavigate();
  const phase = useContext(PilotPhaseContext);
  const PREV = `/income`;
  const NEXT = `/deductions`;

  if (!phase.enableScreener) navigate(`/`);

  return (
    <>
      <Helmet>
        <title>{t(`pages.ScreenerSavingsAndRetirement.pageTitle`)}</title>
      </Helmet>
      <SubHeader />
      <Breadcrumbs href={PREV} />
      <StepIndicator currentStepKey='ScreenerSavingsAndRetirement' />
      <Prose>
        <Heading>{t(`pages.ScreenerSavingsAndRetirement.savingsHeading`)}</Heading>
        <Heading customClassName='savingsAndRetirement'>{t(`pages.ScreenerSavingsAndRetirement.heading`)}</Heading>
        <CommonContentDisplay i18nKey='pages.ScreenerSavingsAndRetirement' TranslationComponent={Translation} />
        <DFIconList>
          <DFIconListItem mood='good' i18nKey={`pages.ScreenerSavingsAndRetirement.savingsList.earnedInterest`} />
          <DFIconListItem mood='good' i18nKey={`pages.ScreenerSavingsAndRetirement.savingsList.addedHSA`} />
          <DFIconListItem mood='good' i18nKey={`pages.ScreenerSavingsAndRetirement.savingsList.flexibleHSA`} />
        </DFIconList>
        <CommonContentDisplay i18nKey='pages.ScreenerSavingsAndRetirement.MSA' TranslationComponent={Translation} />
        <DFIconList>
          <DFIconListItem mood='bad' i18nKey={`pages.ScreenerSavingsAndRetirement.MSA.MSAList.msa`} />
        </DFIconList>
        <DFAccordion i18nKey={`pages.ScreenerSavingsAndRetirement.hsaAccordion`} />
        <Heading customClassName='savingsAndRetirement'>
          {t(`pages.ScreenerSavingsAndRetirement.retirementHeading`)}
        </Heading>
        <CommonContentDisplay
          i18nKey='pages.ScreenerSavingsAndRetirement.retirement'
          TranslationComponent={Translation}
        />
        <DFIconList>
          <DFIconListItem mood='good' i18nKey={`pages.ScreenerSavingsAndRetirement.retirementGoodList.contributions`} />
          <DFIconListItem mood='good' i18nKey={`pages.ScreenerSavingsAndRetirement.retirementGoodList.rollovers`} />
        </DFIconList>
        <CommonContentDisplay
          i18nKey='pages.ScreenerSavingsAndRetirement.retirementNotEligible'
          TranslationComponent={Translation}
        />
        <DFIconList>
          <DFIconListItem
            mood='bad'
            i18nKey={`pages.ScreenerSavingsAndRetirement.retirementNotEligibleList.contributions`}
          />
          <DFIconListItem
            mood='bad'
            i18nKey={`pages.ScreenerSavingsAndRetirement.retirementNotEligibleList.distributions`}
          />
        </DFIconList>
        <DFAccordion i18nKey={`pages.ScreenerSavingsAndRetirement.form1099accordion`} />
        <CommonContentDisplay i18nKey='pages.ScreenerSavingsAndRetirement.link' TranslationComponent={Translation} />
      </Prose>
      <NextLink href={NEXT}>{t(`pages.ScreenerIncome.button.text`)}</NextLink>
    </>
  );
};

export default Retirement;
