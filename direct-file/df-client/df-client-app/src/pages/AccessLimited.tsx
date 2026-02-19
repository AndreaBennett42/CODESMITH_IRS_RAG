import ContentDisplay from '../components/ContentDisplay/index.js';
import PageTitle from '../components/PageTitle/index.js';
import IconDisplay from '../components/IconDisplay/IconDisplay.js';
import { useTranslation } from 'react-i18next';
import styles from './AccessLimited.module.scss';
import useTranslatePIIRedacted from '../hooks/useTranslatePIIRedacted.js';

const AccessLimited = () => {
  const { t } = useTranslation();
  const redacted = useTranslatePIIRedacted(`pages.accessLimited.header`, true);

  return (
    <>
      <div className={styles.dfPageIcon}>
        <IconDisplay name='InfoOutline' size={9} isCentered />
      </div>
      <span className='screen__header'>
        <PageTitle redactedTitle={redacted} large>
          {t(`pages.accessLimited.header`)}
        </PageTitle>
      </span>
      <div className='usa-prose'>
        <ContentDisplay i18nKey={`pages.accessLimited`} allowedTags={[`p`, `h2`]} />
      </div>
    </>
  );
};

export default AccessLimited;
