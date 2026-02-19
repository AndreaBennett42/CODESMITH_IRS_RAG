import { useTranslation } from 'react-i18next';
import * as sfg from '@irs/js-factgraph-scala';
import useTranslationContextFromFacts from './useTranslationContextFromFacts.js';
import { I18nKey } from '../components/Translation/Translation.js';

// This hook returns a function that takes a path and collectionId
// and returns the resulting string interpolated with facts
// Note: Recommend memoizing this function at time of use for performance
const useTranslateWithFacts = (factGraph: sfg.FactGraph, collectionId: string | null = null) => {
  const { t: translate } = useTranslation(`translation`);
  const getContext = useTranslationContextFromFacts(factGraph, collectionId);

  const t = (path: string | string[]): string => {
    const uninterpolatedTrans = translate(path);
    const context = getContext(path);
    // Return a translated version
    return translate(uninterpolatedTrans, context);
  };

  const contextHasData = (paths: I18nKey): boolean => {
    paths = Array.isArray(paths) ? paths : [paths];
    return paths.every((path) => {
      const context = getContext(path);
      return Object.keys(context).every((key) => context[key as keyof typeof context] !== ``);
    });
  };

  return { t, contextHasData };
};

export default useTranslateWithFacts;
