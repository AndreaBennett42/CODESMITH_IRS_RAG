export const LANGUAGE_CODE_PLACEHOLDER = `{LANGUAGE_CODE}`;

export const urlHasLanguagePlaceholder = (urlString: string): boolean => urlString.includes(LANGUAGE_CODE_PLACEHOLDER);

/**
 *
 * @param urlString
 * @param languageCode Potentially an external systems language identifier. Might be `en`, `en-US`, `english`, etc.
 */
export const getTranslatedLink = (urlString: string, languageCode: string) => {
  return urlString.replace(LANGUAGE_CODE_PLACEHOLDER, languageCode);
};
