import { TFunction, type i18n } from 'i18next';
import { getNamespacedKey } from './getNamespacedKey.js';

export type GenericLink = {
  body: object;
  urls?: { [linkKey: string]: string };
};

export type HelpLinkData = {
  text: string;
  urls: { [linkKey: string]: string };
};

export type ModalEntry = {
  header: string;
} & GenericLink;

export type ModalData = {
  text: string;
} & {
  [key: string]: ModalEntry;
};

export type DataShape = HelpLinkData | ModalData | GenericLink;

const isHelpLinkData = (data: unknown): data is HelpLinkData => {
  return data !== null && typeof data === `object` && `text` in data && `urls` in data;
};

const isGenericLink = (data: unknown): data is GenericLink => {
  return data !== null && typeof data === `object` && `body` in data;
};

const isModalData = (data: unknown): data is ModalData => {
  return (
    data !== null &&
    typeof data === `object` &&
    `text` in data &&
    Object.keys(data).some((key) => key.startsWith(`LinkModal`))
  );
};

export const getTranslationKey = (key: string | string[], data: DataShape): string | string[] => {
  if (typeof data === `object`) {
    return isGenericLink(data) ? `${key}.body` : `${key}.text`;
  } else {
    return key;
  }
};

export const maybeUrls = (t: TFunction, i18nKey: string | string[]) => {
  const data = t(i18nKey, { returnObjects: true }) as DataShape;
  let urls: Record<string, string> = {};
  if (isHelpLinkData(data) || isGenericLink(data)) {
    urls = data.urls ?? {};
  } else {
    const linkModalPattern = /LinkModal(\d+)\.body\.text$/;
    let modalMatch: RegExpMatchArray | null = null;
    if (typeof i18nKey === `string`) {
      modalMatch = (i18nKey as string).match(linkModalPattern);
    }
    if (modalMatch) {
      /* Translation calls this function for each snippet of the yaml, and we need to associate the url
      with the correct snippet.  But we only have access to the modal object at the top level key, and not
      when we're iterating over the individual modal, so we drill down to the LinkModal sections,
      then reconstruct the top level key to retrieve the data and extract the urls */
      const modalNum = modalMatch[1];

      const rootModalKey = (i18nKey as string).slice(0, -`.LinkModal${modalNum}.body.text`.length);
      const modalData = t(rootModalKey, {
        returnObjects: true,
      }) as DataShape;

      if (isModalData(modalData)) {
        const linkModalKey = `LinkModal${modalNum}`;

        if (linkModalKey in modalData && `urls` in modalData[linkModalKey]) {
          urls = modalData[linkModalKey].urls ?? {};
        }
      }
    }
  }

  return { data, urls };
};

/**
 * Given an array of keys and a reference to `i18n`, returns the first key
 * that exists. This is good for setting defaults and fallbacks.
 */
export function getFallbackKey(key: string[], i18n: i18n): string | null {
  const allKeys = key.map((key) => getNamespacedKey(key)).filter((key) => !!key);
  // find and return the first good key
  return allKeys.find((possibleKey) => i18n.exists(possibleKey)) ?? null;
}
