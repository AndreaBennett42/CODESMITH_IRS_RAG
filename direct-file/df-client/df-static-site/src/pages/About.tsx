import { useContext } from 'react';
import { CommonContentDisplay } from '@irs/df-common';
import { Helmet } from 'react-helmet-async';
import { Translation } from '../components/index.js';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { PilotPhaseContext } from '../layouts/Providers.js';
import styles from './About.module.scss';

const About = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const phase = useContext(PilotPhaseContext);
  if (!phase.enableScreener) navigate(`/`);
  return (
    <>
      <Helmet>
        <title>{t(`pages.About.pageTitle`)}</title>
      </Helmet>
      <header>
        <h1 className={`screen-heading--large ${styles.about}`}>{t(`pages.About.pageTitle`)}</h1>
      </header>
      <div className='usa-prose'>
        <CommonContentDisplay i18nKey={`pages.About`} allowedTags={[`p`, `h2`]} TranslationComponent={Translation} />
      </div>
    </>
  );
};

export default About;
