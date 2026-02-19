import { useTranslation } from 'react-i18next';
import { useLocation, useSearchParams } from 'react-router-dom';
import { SwitchLangFunc } from './GlobalLayout.js';
import { useEffect } from 'react';

/** This whole component is basically a side effect to set the language */
export const AutoLanguageSelector = ({ switchLang }: { switchLang: SwitchLangFunc }) => {
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const { i18n } = useTranslation(`translation`);

  const urlLang = location.pathname.slice(1).split(`/`)[0];
  const querySetsSpanish = searchParams.get(`forceSpanish`) === `true`;

  // Stored and/or browser lang is loaded on initialization.
  // But in the client we also want to override using the query or url
  useEffect(() => {
    if (urlLang === `es` || querySetsSpanish) {
      if (i18n.language === `es-US`) return; // avoid recursive render
      switchLang(`es-US`, querySetsSpanish);
    }
  }, [urlLang, querySetsSpanish, switchLang, i18n.language]);

  return null;
};
