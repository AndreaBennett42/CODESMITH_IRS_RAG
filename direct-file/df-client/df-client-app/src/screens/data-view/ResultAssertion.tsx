import { useTranslation } from 'react-i18next';
import Translation from '../../components/Translation/index.js';
import { Link } from 'react-router-dom';

export type ResultAssertionProps = {
  asHeading?: boolean;
  reviewRoute?: string;
  i18nKey: string;
  collectionId: string | null;
};

const ResultAssertion = ({ asHeading = false, i18nKey, collectionId, reviewRoute }: ResultAssertionProps) => {
  const { t } = useTranslation();
  return (
    <>
      {asHeading ? (
        <div className='tablet:display-flex tablet:flex-justify '>
          <h2 id={t(`dataviews.result`)}>
            <Translation i18nKey={`dataviews.result`} collectionId={collectionId} />
          </h2>
          {reviewRoute && (
            <Link className='padding-y-4' to={reviewRoute}>
              {t(`button.review`)}
              <span className='usa-sr-only sr-firefox-fix'>
                <Translation i18nKey={`dataviews.result`} collectionId={collectionId} />
              </span>
            </Link>
          )}
        </div>
      ) : (
        <strong className='display-block'>
          <Translation i18nKey={`dataviews.result`} collectionId={collectionId} />
        </strong>
      )}
      <Translation i18nKey={i18nKey} collectionId={collectionId} />
    </>
  );
};

export default ResultAssertion;
