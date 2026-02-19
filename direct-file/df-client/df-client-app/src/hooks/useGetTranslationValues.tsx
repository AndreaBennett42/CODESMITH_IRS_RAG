import { useTranslation } from 'react-i18next';
import { useCallback } from 'react';
import { getTranslationValuesRecursive } from '../utils/i18nUtils.js';

/**
 * Returns a function that gets all the string i18n key values at the supplied key.
 * A potentially useful utility packaged as a hook to encapsulate the minor overhead of composing and using
 * useTranslation.
 */
export const useGetTranslationValues = () => {
  const { i18n, t } = useTranslation();

  return useCallback(
    (i18nKey: string): string[] => {
      const exists = i18n.exists(i18nKey);
      return exists ? getTranslationValuesRecursive(t(i18nKey, { returnObjects: true })) : [];
    },
    [i18n, t]
  );
};
