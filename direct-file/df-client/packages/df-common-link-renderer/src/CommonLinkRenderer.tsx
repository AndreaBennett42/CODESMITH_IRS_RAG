import * as React from 'react';
import { Link } from 'react-router-dom';
import { isValidURL, isExternalLink } from './commonLinkUtilities.js';
import { ModalOpenLink, ModalRef, Icon } from '@trussworks/react-uswds';
import classNames from 'classnames';

type CommonLinkRendererProps = React.AnchorHTMLAttributes<HTMLAnchorElement> & {
  url: string;
  modalRef?: React.RefObject<ModalRef>;
  modalOpenerClasses?: string;
  icon?: Exclude<keyof typeof Icon, 'prototype'>;
};

export const CommonLinkRenderer: React.FC<CommonLinkRendererProps> = ({
  children,
  url,
  modalOpenerClasses,
  modalRef,
  className,
  icon,
  ...props
}) => {
  if (modalRef) {
    const IconComponent = icon ? Icon[icon] : undefined;
    const linkClasses = classNames(modalOpenerClasses, {
      'display-flex flex-align-center': IconComponent,
    });
    return (
      <ModalOpenLink variant='unstyled' href='#info' modalRef={modalRef} {...props} className={linkClasses}>
        {IconComponent && <IconComponent size={3} aria-hidden='true' className='margin-right-05' />}
        {children}
      </ModalOpenLink>
    );
  }

  if (url && isValidURL(url)) {
    const external = isExternalLink(url);
    const linkClasses = classNames(className, {
      'usa-link--external': external,
    });
    const referrer = external ? `noreferrer` : ``;
    const target = external ? `_blank` : ``;

    return (
      <a className={linkClasses} rel={referrer} target={target} href={url} {...props} onClick={() => {}}>
        {children}
      </a>
    );
  }

  const isInternalLink = (url: string): boolean => !/^https?:\/\//.test(url);
  const isImageLink = (url: string): boolean => /\.(jpeg|jpg|gif|png)$/.test(url);

  if (isImageLink(url)) {
    return <img src={`${import.meta.env.VITE_PUBLIC_PATH}/imgs/${url}`} alt={children as string} />;
  }

  if (isInternalLink(url)) {
    return (
      <Link className={className} to={url}>
        {children}
      </Link>
    );
  }

  return <>{children}</>;
};
