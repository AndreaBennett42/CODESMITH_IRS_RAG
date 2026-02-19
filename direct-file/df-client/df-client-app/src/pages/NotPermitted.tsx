import ContentDisplay from '../components/ContentDisplay/index.js';
import PageTitle from '../components/PageTitle/index.js';
import IconDisplay from '../components/IconDisplay/IconDisplay.js';
import { useTranslation } from 'react-i18next';
import styles from './NotPermitted.module.scss';
import useTranslatePIIRedacted from '../hooks/useTranslatePIIRedacted.js';

const NotPermitted = () => {
  const { t } = useTranslation();
  const redacted = useTranslatePIIRedacted(`pages.notPermitted.header`, true);

  return (
    <>
      <div className={styles.dfPageIcon}>
        <IconDisplay name='InfoOutline' size={9} isCentered />
      </div>
      <span className='screen__header'>
        <PageTitle redactedTitle={redacted} large>
          {t(`pages.notPermitted.header`)}
        </PageTitle>
      </span>
      <div className='usa-prose'>
        <ContentDisplay i18nKey={`pages.notPermitted`} allowedTags={[`p`, `h2`]} />
      </div>
    </>
  );
};

export default NotPermitted;
