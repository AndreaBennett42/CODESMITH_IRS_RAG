import { TFunction } from 'i18next';
import { ScreenContentConfig } from '../flow/ContentDeclarations.js';
import { InfoDisplayProps } from '../types/core.js';
import { hasOwn } from './polyfills.js';

/**
 * Recursively checks an i18n returned value for a given translation key.
 * Typically, this is a string, but if a parent node/key is requested it can be a complex object or an array
 * @param value the i18n returned value
 */
export const getTranslationValuesRecursive = (value: string | object): string[] => {
  const values: string[] = [];
  if (typeof value === `string`) {
    // it's a string
    values.push(value);
  }
  if (typeof value === `object`) {
    if (Array.isArray(value)) {
      // it's an array
      values.push(...value.flatMap((arrayChild) => getTranslationValuesRecursive(arrayChild)));
    } else {
      // it's an object
      values.push(...Object.values(value).flatMap((childValue) => getTranslationValuesRecursive(childValue)));
    }
  }
  return values;
};

export function findComponentsOfType<ComponentName extends ScreenContentConfig[`componentName`]>(
  targetComponentName: ComponentName,
  componentList: ScreenContentConfig[]
): ScreenContentConfig[] {
  // eslint-disable-next-line eqeqeq
  const results = componentList.filter((config) => config.componentName == targetComponentName);
  return results;
}

export const getStringKey = (key: string | undefined, t: TFunction) => {
  if (!key) {
    return key;
  }
  // Sometimes the key holds a modal body
  const data = t(key, { returnObjects: true });
  const isModal = checkIfModal(data);
  // If so, we modify the key to access the simple string title
  key = isModal ? `${key}.helpText.modals.text` : key;
  return key;
};

export const getTitleKey = (screen: ScreenContentConfig[]) => {
  const headings = findComponentsOfType(`Heading`, screen);
  if (headings.length === 0) {
    return undefined;
  } else {
    const headingProps = headings[0].props as InfoDisplayProps;
    return `headings.${headingProps.i18nKey}`;
  }
};

export const checkIfModal = (data: object) => {
  return data && !checkIfHasInternalLink(data) && typeof data === `object` && hasOwn(data, `helpText`);
};

export const checkIfHasInternalLink = (data: object) => {
  return data && hasOwn(data, `internalLink`);
};
