import { useTranslation } from 'react-i18next';
import Translation from '../Translation/index.js';
import { Link, useLocation } from 'react-router-dom';
import cn from 'classnames';

import { CommonHeader } from '@irs/df-common';
import { LanguageSelector } from '../LanguageSelector/LanguageSelector.js';
import { useEffect, useRef } from 'react';

import { isSpanishEnabled } from '../../constants/pageConstants.js';
import { Modal, ModalHeading, ModalRef } from '@trussworks/react-uswds';
import ContentDisplay from '../ContentDisplay/index.js';
import { SwitchLangFunc } from '../GlobalLayout.js';

type HeaderProps = {
  switchLang: SwitchLangFunc;
  autoSpanishModal: boolean; // allows parent to force pop the modal
};

const Header = ({ switchLang, autoSpanishModal: showSpanishModal }: HeaderProps) => {
  const location = useLocation();
  const { t, i18n } = useTranslation(`translation`);
  const now = new Date();

  const modalRef = useRef<ModalRef>(null);

  useEffect(() => {
    if (showSpanishModal) {
      modalRef.current?.toggleModal(undefined, true);
    }
  }, [showSpanishModal]);

  const isCurrentPath = (path: string) => {
    const normalizedPath = location.pathname.endsWith(`/`) ? location.pathname : location.pathname + `/`;
    return normalizedPath === path;
  };

  const menuItems = [
    <Link
      className={cn(`usa-nav__link`, { 'usa-current': isCurrentPath(`/account/`) })}
      to='/account/'
      key='account'
      aria-current={isCurrentPath(`/account/`) ? `page` : undefined}
    >
      {t(`account.title`)}
    </Link>,
  ];

  const onClickLanguageChange = (changeLang: string) => {
    if (changeLang === `es-US` && !isSpanishEnabled()) {
      return () => modalRef.current?.toggleModal(undefined, true);
    } else {
      return () => switchLang(changeLang);
    }
  };

  const languageOptions = [
    {
      label: `English`,
      attr: `en`,
      on_click: onClickLanguageChange(`en`),
    },
    {
      label: `Español`,
      label_local: `Spanish`,
      attr: `es-US`,
      on_click: onClickLanguageChange(`es-US`),
    },
  ];

  // Set the button displayLang to the other language
  let displayLang = `es-US`;
  if (i18n.language === `es-US`) {
    displayLang = `en`;
  }
  return (
    <CommonHeader
      headerAriaLabel={t(`header.banner-label`)}
      dfPrefix={`${import.meta.env.VITE_PUBLIC_PATH}`}
      homeLink='/home'
      homeLinkTitle='header.title-link-label'
      navMenuLabel={t(`header.menu`)}
      menuItems={menuItems}
    >
      <LanguageSelector langs={languageOptions} displayLang={displayLang} />
      <Modal
        id='spanish-unavailable-modal'
        ref={modalRef}
        aria-labelledby='spanish-unavailable-heading'
        aria-describedby='spanish-unavailable-description'
      >
        <ModalHeading id={`spanish-unavailable-heading`}>
          <Translation collectionId={null} i18nKey={`header.no-spanish-modal.header`} />
        </ModalHeading>
        <div className='usa-prose' id='spanish-unavailable-description'>
          <ContentDisplay collectionId={null} i18nKey={`header.no-spanish-modal`} />
        </div>
      </Modal>
    </CommonHeader>
  );
};

export default Header;
