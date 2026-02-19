import { useEffect, ReactNode, FC } from 'react';
import { useLocation } from 'react-router-dom';
import classNames from 'classnames';
import { Helmet } from 'react-helmet-async';

export const PAGE_HEADING_ID = `page-heading`;

const PageTitle: FC<{
  children: string | ReactNode;
  large?: boolean;
  redactedTitle: string;
}> = (props) => {
  const location = useLocation();

  const headingClasses = classNames(`screen-heading`, { 'screen-heading--large': props.large });
  useEffect(() => {
    const landingPages = [`/home`, `/login`];
    const onLandingPage = landingPages.includes(location.pathname);
    if (!onLandingPage) {
      document.getElementById(`main`)?.focus();
      window.scrollTo(0, 0);
    }
  }, [location, props.redactedTitle]);

  return (
    <>
      <Helmet title={props.redactedTitle} />
      <h1 className={headingClasses} id={PAGE_HEADING_ID}>
        {props.children}
      </h1>
    </>
  );
};

export default PageTitle;
