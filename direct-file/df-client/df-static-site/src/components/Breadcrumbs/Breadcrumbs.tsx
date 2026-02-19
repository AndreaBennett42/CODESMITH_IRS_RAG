import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import { Icon } from '@trussworks/react-uswds';

const Breadcrumbs = ({ href }: { href: string }) => {
  const { t } = useTranslation(`translation`);

  return (
    <div className='margin-top-1 margin-bottom-3'>
      <Link to={href}>
        <Icon.ArrowBack className='top-05' aria-hidden='true' />
        <span className='margin-left-05'>{t(`components.breadcrumbs.back`)}</span>
      </Link>
    </div>
  );
};

export default Breadcrumbs;
