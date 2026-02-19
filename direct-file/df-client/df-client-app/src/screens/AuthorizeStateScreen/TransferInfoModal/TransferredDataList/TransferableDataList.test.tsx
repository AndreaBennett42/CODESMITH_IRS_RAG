import TransferableDataList from './TransferableDataList.js';
import { exportToStateFacts } from '../../../../fact-dictionary/generated/exportToStateFacts.js';
import { Path } from '../../../../flow/Path.js';
import { COLLECTION_INDICATOR, getFactKeyFromPath } from '../../../../utils/exportUtils.js';
import { i18n as i18nType } from 'i18next';
import { initI18n } from '../../../../i18n.js';
import { getCollectionLabelI18nKey, getFactLabelI18nKey } from './utils.js';

describe(TransferableDataList.name, () => {
  describe(`label translations`, () => {
    let i18n: i18nType;

    beforeAll(async () => {
      i18n = await initI18n();
    });

    const getMissingTranslationKeysForLanguage = (lng: string) => {
      const missingKeys = new Set<string>();

      // every exportable fact path must have a corresponding entry in the translation files
      exportToStateFacts.forEach((exportableFactPath) => {
        if (Path.isAbstractPathType(exportableFactPath)) {
          // an abstract path fact must have an entry for the collection label as well as the fact
          const collectionIndicatorIndex = exportableFactPath.indexOf(COLLECTION_INDICATOR);
          const collectionKey = exportableFactPath.substring(1, collectionIndicatorIndex);
          const collectionLabelI18nKey = getCollectionLabelI18nKey(collectionKey);
          if (!i18n.exists(collectionLabelI18nKey, { lng }) && !missingKeys.has(collectionLabelI18nKey)) {
            // collection label was missing
            missingKeys.add(collectionLabelI18nKey);
          }

          const factKey = getFactKeyFromPath(exportableFactPath);
          const factLabelI18nKey = getFactLabelI18nKey(factKey, collectionKey);
          if (!i18n.exists(factLabelI18nKey, { lng }) && !missingKeys.has(factLabelI18nKey)) {
            // collection fact label was missing
            missingKeys.add(factLabelI18nKey);
          }
        } else {
          // Add the fact key if not already added
          const factKey = getFactKeyFromPath(exportableFactPath);
          const factLabelI18nKey = getFactLabelI18nKey(factKey);
          if (!missingKeys.has(factLabelI18nKey)) {
            // fact label was missing
            missingKeys.add(factLabelI18nKey);
          }
        }
      });

      return missingKeys;
    };

    it.each([`en`, `es`])(`has %s translations for every exportable field`, (languageCode) => {
      // set up object to cache all the missing keys for the eventual assertion
      const missingKeys = getMissingTranslationKeysForLanguage(languageCode);

      try {
        expect(missingKeys.size).toEqual(0);
      } catch (e) {
        const missingKeysArray = Array.from(missingKeys);
        throw new Error(
          `Labels for exported data disclosure are missing translation keys.
          The following ${missingKeys.size} translation key(s) were expected but not found in ${languageCode}.yaml:
          - ${missingKeysArray.join(`,\n          - `)}`
        );
      }
    });
  });

  it.todo(`Render related tests`);
});
