import classNames from 'classnames';
import Translation from '../../components/Translation/index.js';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

export type AssertionProps = {
  i18nKey: string;
  type: `success` | `warning` | `inactive` | `info`;
  collectionId: string | null;
  editRoute?: string;
};

const Assertion = ({ i18nKey, type, collectionId, editRoute }: AssertionProps) => {
  const { t } = useTranslation();
  const assertionClass = classNames({
    'usa-alert--success': type === `success`,
    'usa-alert--warning': type === `warning`,
    'bg-base-lightest': type === `inactive`,
    'usa-alert--info': type === `info`,
  });
  return (
    <div className={`margin-top-3 usa-alert usa-alert--no-icon ${assertionClass}`}>
      <div className='usa-alert__body'>
        <Translation i18nKey={i18nKey} collectionId={collectionId} />
        &nbsp;
        {editRoute && <Link to={editRoute}>{t(`button.edit`)}</Link>}
      </div>
    </div>
  );
};

export default Assertion;
