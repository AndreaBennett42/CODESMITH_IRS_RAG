import ContentDisplay from '../components/ContentDisplay/index.js';
import PageTitle from '../components/PageTitle/index.js';
import { useTranslation } from 'react-i18next';
import useTranslatePIIRedacted from '../hooks/useTranslatePIIRedacted.js';

export const About = () => {
  const { t } = useTranslation();
  const redacted = useTranslatePIIRedacted(`about.header`, true);

  return (
    <>
      <span className='screen__header'>
        <PageTitle redactedTitle={redacted} large>
          {t(`about.header`)}
        </PageTitle>
      </span>
      <div className='usa-prose'>
        <ContentDisplay i18nKey={`about`} allowedTags={[`p`, `h2`]} />
      </div>
    </>
  );
};
