import { useCallback } from 'react';
import { useGetTranslationValues } from './useGetTranslationValues.js';
import { pathReferencePattern } from './useTranslationContextFromFacts.js';

/**
 * Returns a function that returns true if the i18n-returned value at the supplied key (or any of its children) contains
 * a fact path
 */
export const useContainsFactGraphValues = () => {
  const getTranslationValues = useGetTranslationValues();

  return useCallback(
    (i18nKey: string): boolean => {
      const translationValues = getTranslationValues(i18nKey);
      return translationValues.some((value) => value.match(pathReferencePattern));
    },
    [getTranslationValues]
  );
};
