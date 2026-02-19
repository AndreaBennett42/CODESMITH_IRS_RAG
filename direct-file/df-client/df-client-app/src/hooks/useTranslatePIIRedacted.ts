import { useTranslation } from 'react-i18next';
import { TFunction } from 'i18next';
import { CURRENT_TAX_YEAR, DEFAULT_TAX_DAY } from '../constants/taxConstants.js';
export const variablePattern = /{{(.+?)}}/g;

// Path key must resolve to a string translation, not a complex object
export function redactPII(t: TFunction, path: string, stripHTMLTags: boolean, redactedContext?: object) {
  const uninterpolatedTrans: string = t(path);
  const rawPaths = [...uninterpolatedTrans.matchAll(variablePattern)];

  const redactedValue = `***`;
  // For each match, convert it into pairs like ['/filingStatus', '***']
  const variableData = rawPaths.map((matchGroup) => {
    return [matchGroup[1], redactedValue];
  });

  // Create a context object with all the resultsm
  // Add replacement values for any non-PII facts that are allowed
  const context = {
    ...Object.fromEntries(variableData),
    // TODO: We probably want to add some limited factgraph access but for now
    // to ensure we protect PII we are avoiding giving this function fg access.
    // https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/5058
    '/taxYear': `${CURRENT_TAX_YEAR}`,
    '/lastTaxYear': `${parseInt(CURRENT_TAX_YEAR) - 1}`,
    '/nextTaxYear': `${parseInt(CURRENT_TAX_YEAR) + 1}`,
    '/defaultTaxDay': `${DEFAULT_TAX_DAY}`,
    ...redactedContext,
  };
  // Create a redacted translation
  const redacted = t(path, context) as string;
  if (stripHTMLTags) {
    // Strip all HTML tags from redacted but keep innertext
    return redacted.replace(/<[^>]*>/g, ``);
  } else {
    return redacted;
  }
}
// This function does not and should not have access to the fact graph
// as it is used to redact PII from translations.
// The path will be translated and all facts and variables replaced with ***
// You may pass in some context but ensure that it does not contain PII
// You may also whitelist some strings in this function, again no PII
// This function also optionally strips tags
const useTranslatePIIRedacted = (path: string, stripHTMLTags: boolean, redactedContext?: object) => {
  // redactPII was extracted to allow more flexible usage.
  // However, leaving this hook because we want to add more factgraph functionality here soon.
  const { t } = useTranslation();
  return redactPII(t, path, stripHTMLTags, redactedContext);
};

export default useTranslatePIIRedacted;
