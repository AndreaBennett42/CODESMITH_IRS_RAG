import { TranslationProps } from '../CommonContentDisplay/contentGenerator.js';
import { FunctionComponent } from 'react';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import { useTranslation } from 'react-i18next';
import { CommonContentDisplay } from '../CommonContentDisplay/CommonContentDisplay.js';

type CommonNotFoundProps = {
  i18nKey: string;
  TranslationComponent: FunctionComponent<TranslationProps>;
};

const CommonNotFound: FunctionComponent<CommonNotFoundProps> = ({ i18nKey, TranslationComponent }) => {
  const { i18n } = useTranslation(`translation`);
  const allowedTags = [`h1`, `h2`, `p`];
  return (
    <div className='df-not-found'>
      <CommonContentDisplay
        i18nKey={i18nKey}
        collectionId={null}
        allowedTags={allowedTags}
        TranslationComponent={TranslationComponent}
      />
      <TranslationComponent
        i18nKey={`notFoundDashboardLink`}
        collectionId={null}
        components={{
          InternalLink: <CommonLinkRenderer url={`/home`} />,
        }}
      />
      <p className='text-base, margin-y-5'>
        <strong>Error code:</strong> 404
      </p>
    </div>
  );
};

export default CommonNotFound;
