import { useTranslation } from 'react-i18next';
import * as sfg from '@irs/js-factgraph-scala';
import { useMemo } from 'react';
import { Path } from '../flow/Path.js';
import { Path as FGPath } from '../fact-dictionary/Path.js';
import { prettyPrintFactGraphValues } from '../misc/factGraphPrettyPrint.js';

export const pathReferencePattern = /{{(\/[^, }]+)([^\]]*?)}}/g;

// Returns a function that takes a string and collection info and returns
// an object with all the variables from the string and the resulting facts from the factgraph
const useTranslationContextFromFacts = (factGraph: sfg.FactGraph, collectionId: string | null = null) => {
  const { t } = useTranslation(`translation`);
  return useMemo(() => {
    return (path: string | string[]): Partial<Record<FGPath, string | Date | number>> => {
      // Disable interpolation by setting a fake prefix; set to empty string if translation not found
      const uninterpolatedTrans = t(path, { interpolation: { prefix: `NO INTERPOLATION` } }) || ``;
      // Extract the specified path, ignoring any formatting options
      // Find variables in the uninterpolated string
      const rawPaths = [...uninterpolatedTrans.matchAll(pathReferencePattern)];

      // For each variable, find the value from factgraph.
      if (rawPaths.length) {
        // Default format parameters to apply
        const formatParams: Partial<Record<FGPath, unknown>> = {};

        // For each match, convert it into pairs like ['/filingStatus', 'single']
        const variableData = rawPaths.map((matchGroup) => {
          const rawPath = matchGroup[1] as FGPath;

          // Special case that we may have an empty collection, and we do not want to error out.
          // https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/1093
          // eslint-disable-next-line eqeqeq
          if (rawPath.indexOf(`*`) != -1 && collectionId == undefined) {
            return [rawPath, ``];
          }
          const concretePath = Path.concretePath(rawPath, collectionId);
          const data = factGraph.get(concretePath);
          const { output, formatParams: localFormatParams } = prettyPrintFactGraphValues(rawPath, data, t, `fields`);
          Object.entries(localFormatParams).forEach((kv) => {
            formatParams[kv[0] as FGPath] = kv[1];
          });
          return [rawPath, output];
        });
        // Create a context object with all the results
        const context = {
          formatParams,
          ...Object.fromEntries(variableData),
        };
        return context;
      }
      return {};
    };
  }, [collectionId, factGraph, t]);
};

export default useTranslationContextFromFacts;
