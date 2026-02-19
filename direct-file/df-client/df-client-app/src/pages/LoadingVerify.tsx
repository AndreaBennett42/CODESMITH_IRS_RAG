import PageTitle from '../components/PageTitle/index.js';
import useTranslatePIIRedacted from '../hooks/useTranslatePIIRedacted.js';
import { useEffect } from 'react';
import LoadingIndicator from '../components/LoadingIndicator/LoadingIndicator.js';
import { useNavigate } from 'react-router-dom';
import styles from './LoadingVerify.module.scss';

// This page is an interstitial that currently does nothing. We should refactor
// the way all of this gets setup to avoid many loading gates.
const LoadingVerify = () => {
  const redacted = useTranslatePIIRedacted(`pages.loadingVerify.message`, true);
  const navigate = useNavigate();

  useEffect(() => {
    navigate(`/home`);
  });

  return (
    <div className={styles.loadingVerify}>
      <LoadingIndicator i18nKey={`pages.loadingVerify.message`} />
      <span className='screen__header'>
        <PageTitle redactedTitle={redacted} large>
          <></>
        </PageTitle>
      </span>
    </div>
  );
};

export default LoadingVerify;
