import { useTranslation } from 'react-i18next';
import { useCallback } from 'react';
import { getTranslatedLink, LANGUAGE_CODE_PLACEHOLDER, urlHasLanguagePlaceholder } from '../utils/urlUtils.js';
import { StateProfile } from '../types/StateProfile.js';

const DEFAULT_LANGUAGE_CODE = `en`;

const useUrlTranslator = () => {
  const { i18n } = useTranslation(`translation`);

  const translateUrl = useCallback(
    (urlString: string, supportedLanguages: Record<string, string>): string => {
      if (urlHasLanguagePlaceholder(urlString)) {
        const i18nLanguageCode = i18n.resolvedLanguage || ``;

        const supportedLanguage = supportedLanguages[i18nLanguageCode];

        if (supportedLanguage) {
          return getTranslatedLink(urlString, supportedLanguage);
        }

        return getTranslatedLink(urlString, DEFAULT_LANGUAGE_CODE);
      }

      return urlString;
    },
    [i18n.resolvedLanguage]
  );

  const translateStateProfileUrls = useCallback(
    (stateProfile: StateProfile): StateProfile => {
      const {
        stateCode,
        landingUrl,
        defaultRedirectUrl,
        transferCancelUrl,
        waitingForAcceptanceCancelUrl,
        redirectUrls,
        languages,
      } = stateProfile;

      const translatedLandingUrl = translateUrl(landingUrl, languages);
      const translatedDefaultRedirectUrl = defaultRedirectUrl && translateUrl(defaultRedirectUrl, languages);
      const translatedTransferCancelUrl = transferCancelUrl && translateUrl(transferCancelUrl, languages);
      const translatedWaitingForAcceptanceCancelUrl =
        waitingForAcceptanceCancelUrl && translateUrl(waitingForAcceptanceCancelUrl, languages);
      const translatedRedirectUrls = redirectUrls.map((redirectUrl) => translateUrl(redirectUrl, languages));

      const stateProfileWithTranslatedUrls: StateProfile = {
        ...stateProfile,
        landingUrl: translatedLandingUrl,
        defaultRedirectUrl: translatedDefaultRedirectUrl,
        transferCancelUrl: translatedTransferCancelUrl,
        waitingForAcceptanceCancelUrl: translatedWaitingForAcceptanceCancelUrl,
        redirectUrls: translatedRedirectUrls,
      };

      return stateProfileWithTranslatedUrls;
    },
    [translateUrl]
  );

  return { translateUrl, translateStateProfileUrls };
};

export default useUrlTranslator;
