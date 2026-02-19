import { useTranslation } from 'react-i18next';
import { useLocation } from 'react-router-dom';
import classnames from 'classnames';

const SubHeader = ({ mobile = false }: { mobile?: boolean }) => {
  const { t } = useTranslation(`translation`);
  const { pathname } = useLocation();
  const classNames = classnames(`site-subheader`, mobile ? `site-subheader--mobile` : `site-subheader--desktop`);

  // eslint-disable-next-line eqeqeq
  if (pathname === `/` || pathname == `/done` || pathname == `/limited` || pathname == `/about/`) return null;

  return (
    <header aria-label={t(`components.subheader.label`)} className={classNames}>
      {t(`components.subheader.title`)}
    </header>
  );
};

export default SubHeader;
