import { pathReferencePattern } from '../hooks/useTranslationContextFromFacts.js';

const logger = console;
export type RawLocale = Record<string, string | object>;
export type LocaleGroup = Record<string, string | object> | object;

export function normalizeLocaleGroup(
  localeGroup: LocaleGroup | LocaleGroup[] | string | null
): LocaleGroup | LocaleGroup[] | string | null {
  if (localeGroup === null || localeGroup === `TRANSLATE ME`) return null;
  if (typeof localeGroup === `string`) return localeGroup;

  if (Array.isArray(localeGroup)) {
    // Strip out any null items and normalize all contents
    const normalizedArray = localeGroup
      .filter((item) => item !== null)
      .map(normalizeLocaleGroup)
      // Strip out any items that reduce to null
      .filter((item): item is string | LocaleGroup => item !== null);

    // Reduce empty arrays to null, so they are stripped out by any parent calls
    if (normalizedArray.length > 0) return normalizedArray;
    else return null;
  }

  if (Object.keys(localeGroup).every((key) => key.match(/^\d+$/g))) {
    // This should actually be array because the keys are all numbers
    return normalizeLocaleGroup(Array.from(Object.values(localeGroup)));
  } else {
    const normalizedKeyValuePairs = Object.entries(localeGroup)
      // Recursively reduce field values
      .map(([key, value]) => [key, normalizeLocaleGroup(value as string | object)])
      // Strip out any values that reduce to null
      .filter(([_, value]) => value !== null);

    if (normalizedKeyValuePairs.length > 0) return Object.fromEntries(normalizedKeyValuePairs) as LocaleGroup;
    else return null;
  }
}

export default class Locale {
  constructor(protected rawLocale: RawLocale) {}

  #keys?: Set<string> = undefined;
  #reverseMap?: Map<string, Set<string>>;

  get keys() {
    if (undefined === this.#keys) {
      this.#keys = new Set(this.flatten().map(([key]) => key));
    }

    return this.#keys;
  }

  has(i18nKey: string) {
    return this.keys.has(i18nKey) || !!this.get(i18nKey);
  }

  get(i18nKey: string) {
    return i18nKey
      .split(`.`)
      .reduce(
        (localeGroup, keyPart) => (localeGroup as Record<string, string | object>)?.[keyPart],
        this.rawLocale as string | object
      );
  }

  set(i18nKey: string, value: string) {
    if (this.#keys) {
      // Keep the key cache up-to-date, if initialized
      this.#keys.add(i18nKey);
    }

    const keyPath = i18nKey.split(`.`);
    const isNum = (str: string) => {
      return parseInt(str, 10).toString() === str;
    };
    // Follow the key path, creating subjects along the way as needed
    keyPath.reduce((localeGroup, keyPart, index, keyPath) => {
      // Figure out if current key is a leaf, array or object
      const atLeaf = index === keyPath.length - 1;
      const atArray = !atLeaf && isNum(keyPath[index + 1]);
      const atObject = !atLeaf && !atArray;

      // If you are at a leaf just set a value
      if (atLeaf) {
        localeGroup[keyPart] = value;
        return {}; // Return an object to appease the typing gods
      } else {
        // If we're at an array and there's not an array in the destination, create one
        // Note: This is destructive if there is something else in the destination!
        if (atArray && !Array.isArray(localeGroup[keyPart])) {
          localeGroup[keyPart] = [];
        }
        // If we're at an object and there's not an object in the destination, create one
        // Note: This is destructive if there is something else in the destination!
        if (atObject && typeof localeGroup[keyPart] !== `object`) {
          localeGroup[keyPart] = {};
        }
        return localeGroup[keyPart] as RawLocale;
      }
    }, this.rawLocale);
  }

  toJSON() {
    return normalizeLocaleGroup(this.rawLocale);
  }

  static flattenRecurse(locale: RawLocale | object | null): [string, string][] {
    if (locale === null) return [];

    return Object.entries(locale).flatMap(([key, valueOrGroup]) =>
      typeof valueOrGroup === `string`
        ? [[key, valueOrGroup]]
        : this.flattenRecurse(valueOrGroup).map(([subKey, value]) => [`${key}.${subKey}`, value] as [string, string])
    );
  }

  flatten() {
    return Locale.flattenRecurse(this.rawLocale);
  }

  // Diffs two locales
  diff(other: Locale) {
    // We flatten the locales to count and match the keys
    const flatThis = this.flatten();
    const flatOther = other.flatten();

    // Get just the keys for filtering
    const keysThis = flatThis.map((tuple) => tuple[0]);
    const keysOther = flatOther.map((tuple) => tuple[0]);

    // Find matching keys
    const inBoth = flatThis.filter((tuple) => keysOther.includes(tuple[0]));

    // Find keys that were in other locale but not this one
    const removed = flatOther.filter((tuple) => !keysThis.includes(tuple[0]));

    // Find keys that were in this locale but not in the other
    const added = flatThis.filter((tuple) => !keysOther.includes(tuple[0]));

    const thisCount = flatThis.length;
    const otherCount = flatOther.length;
    return { thisCount, otherCount, inBoth, added, removed };
  }

  /** Get a map that  */
  getReverseMap() {
    if (undefined === this.#reverseMap) {
      const reverseMap = this.flatten().reduce((result, [key, value]) => {
        const keyList = result.get(value) ?? new Set();

        keyList.add(key);

        result.set(value, keyList);

        return result;
      }, new Map<string, Set<string>>());

      return (this.#reverseMap = reverseMap);
    }

    return this.#reverseMap;
  }
}

export function normalizeCopy(copy: string) {
  return copy.replace(/\s+/g, ` `).trim();
}

export function translateKey(
  key: string,
  enLocale: Locale,
  flattenedEnLocale: [string, string][] | [],
  locale: Locale,
  updatedEs: string,
  lastKnownEn: string
) {
  const currentEs = normalizeCopy((locale.get(key) ?? ``).toString());
  const currentEn = normalizeCopy(enLocale.get(key)?.toString() ?? ``);
  if (currentEn.length === 0) {
    logger.warn(`[warn] Could not find key in English copy: ${key}`);

    if (lastKnownEn.length > 0 && updatedEs.length > 0) {
      const alternateKeys = flattenedEnLocale.filter(([_, en]) => lastKnownEn === normalizeCopy(en ?? ``)) || [];
      if (alternateKeys.length) {
        logger.warn(
          `[warn] Did it move to following key${alternateKeys.length > 1 ? `s` : ``} with matching English copy?`
        );
        logger.warn(`\tApplying Spanish copy to the alternate keys`);
        logger.warn(`\t• ${alternateKeys.join(`\n\t• `)}`);
        for (const [alternateKey] of alternateKeys) locale.set(alternateKey, updatedEs);
      }
    } else {
      return false;
    }
  } else if (updatedEs !== currentEs) {
    if (currentEn !== lastKnownEn) {
      logger.warn(`[warn] English copy for ${key} may have diverged from English`);
    }
    locale.set(key, updatedEs);
  }
}

export function countGrouped(flatLocale: [string, string][]) {
  // Categorize flat results by first key for reporting
  const grouped = flatLocale.reduce((obj: { [index: string]: number }, [key, _]: [string, string]) => {
    const rootKey = key.split(`.`)[0];
    if (!(rootKey in obj)) {
      obj[rootKey] = 1;
    } else {
      obj[rootKey] += 1;
    }
    return obj;
  }, {} as { [index: string]: number });
  return grouped;
}

export const NESTED_TRANSLATION_REGEX = /\$t\(([*\-\w./]+)\)/g;

export function getVariables(uninterpolatedTrans: string) {
  if (!uninterpolatedTrans) return [];
  const matches = [...uninterpolatedTrans.matchAll(pathReferencePattern)];
  return matches.map((match) => match[1]);
}
