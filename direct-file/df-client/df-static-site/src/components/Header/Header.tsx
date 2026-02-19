import { useMemo } from 'react';
import { CommonHeader, CommonLanguageSelector } from '@irs/df-common';
import { useTranslation } from 'react-i18next';
import SubHeader from '../SubHeader/SubHeader.js';
import { baseRouterPath, englishUrl, spanishUrl } from '../../constants.js';

const Header = () => {
  const { t, i18n } = useTranslation(`translation`);

  const directFileUrl = i18n.language?.startsWith(`es`) ? spanishUrl : englishUrl;

  const onClickLanguageChange = (changeLang: string) => {
    return () => {
      i18n.changeLanguage(changeLang);
      document.documentElement.setAttribute(`lang`, changeLang);
      localStorage.setItem(`irs_df_language`, changeLang);
    };
  };

  const languageOptions = {
    en: {
      label: `English`,
      attr: `en`,
      on_click: onClickLanguageChange(`en`),
    },
    es: {
      label: `Español`,
      label_local: `Spanish`,
      attr: `es-US`,
      on_click: onClickLanguageChange(`es-US`),
    },
    px: {
      label: `Pseudo`,
      label_local: `Pseudo`,
      attr: `px`,
      on_click: onClickLanguageChange(`px`),
    },
  };

  /** Workaround to stop the language toggle saying Español if we're already in Spanish. */
  // TODO: consider using `displayLang` prop once we're on >5.5.0 of @trussworks/react-uswds
  const languagesLink = useMemo(() => {
    return i18n.language === `en` ? [languageOptions.es, languageOptions.en] : [languageOptions.en, languageOptions.es];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <>
      <CommonHeader
        headerAriaLabel={t(`components.header.banner-label`)}
        homeLink='/'
        homeLinkTitle='components.header.title-link-label'
        navMenuLabel={t(`components.header.menu`)}
        dfPrefix={baseRouterPath}
      >
        <div className='usa-language-container'>
          <a className='usa-button usa-language-container' href={directFileUrl}>
            {t(`components.header.sign-in`)}
          </a>
        </div>
        <div className='padding-bottom-1'>
          <CommonLanguageSelector langs={languagesLink} />
        </div>
      </CommonHeader>
      <SubHeader mobile />
    </>
  );
};

export default Header;
