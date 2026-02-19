import { useContainsFactGraphValues } from './useContainsFactGraphValues.js';
import { renderHook } from '@testing-library/react';
import { initI18n } from '../i18n.js';

const A_STRING = `a string`;
const A_STRING_WITH_A_FACT_PATH = `a string with a {{/factPath}}`;

const WITHOUT_FACT_PATHS_PREFIX = `withoutFactPaths`;
const WITH_FACT_PATHS_PREFIX = `withFactPaths`;

const i18nTranslationsWithoutFactPaths = {
  [WITHOUT_FACT_PATHS_PREFIX]: {
    string: A_STRING,
    arrayOfStrings: [A_STRING, A_STRING],
    complexArray: [A_STRING, { child: A_STRING }],
    complexObject: {
      key1: A_STRING,
      key2: [A_STRING, A_STRING, A_STRING],
      key3: {
        nestedKey1: A_STRING,
        nestedKey2: A_STRING,
        nestedKey3: [A_STRING, A_STRING, A_STRING],
      },
    },
    alert: {
      alertText: {
        heading: A_STRING,
        body: A_STRING,
      },
    },
    alertWithInternalLink: {
      alertText: {
        heading: A_STRING,
        body: A_STRING,
      },
      internalLink: A_STRING,
    },
    alertWithNestedKey: {
      alertText: {
        heading: `$t(${WITHOUT_FACT_PATHS_PREFIX}.nested.key)`,
      },
    },
    nested: {
      key: A_STRING,
    },
  },
};

const i18nTranslationsWithFactPaths = {
  [WITH_FACT_PATHS_PREFIX]: {
    string: A_STRING_WITH_A_FACT_PATH,
    arrayOfStrings: [A_STRING, A_STRING_WITH_A_FACT_PATH],
    complexArray: [A_STRING, { child: A_STRING_WITH_A_FACT_PATH }],
    complexObject: {
      child: {
        grandchild: [A_STRING, A_STRING, A_STRING_WITH_A_FACT_PATH],
      },
    },
    alertWithFactPathHeading: {
      alertText: {
        heading: A_STRING_WITH_A_FACT_PATH,
        body: A_STRING,
      },
    },
    alertWithFactPathBody: {
      alertText: {
        heading: A_STRING,
        body: A_STRING_WITH_A_FACT_PATH,
      },
    },
    alertWithFactPathInternalLink: {
      alertText: {
        heading: A_STRING,
        body: A_STRING,
      },
      internalLink: A_STRING_WITH_A_FACT_PATH,
    },
    alertWithFactPathModal: {
      alertText: {
        body: {
          helpText: {
            modals: {
              text: A_STRING_WITH_A_FACT_PATH,
            },
          },
        },
      },
    },
    alertWithNestedKey: {
      alertText: {
        heading: `$t(${WITH_FACT_PATHS_PREFIX}.nested.keyWithFactPath)`,
      },
    },
    nested: {
      keyWithFactPath: A_STRING_WITH_A_FACT_PATH,
    },
  },
};

const testTranslations = {
  ...i18nTranslationsWithoutFactPaths,
  ...i18nTranslationsWithFactPaths,
};

describe(useContainsFactGraphValues.name, () => {
  beforeAll(async () => {
    const i18nInstance = await initI18n();
    i18nInstance.addResourceBundle(`test`, `translation`, testTranslations);
    await i18nInstance.changeLanguage(`test`);
  });

  it.each(Object.entries(i18nTranslationsWithoutFactPaths[WITHOUT_FACT_PATHS_PREFIX]))(
    `callback returns false if no translation key contains a fact path "${WITHOUT_FACT_PATHS_PREFIX}.%s": %o`,
    (i18nKey, _i18nValue) => {
      const {
        result: { current: containsFactGraphValues },
      } = renderHook(useContainsFactGraphValues);

      const foundFactGraphValues = containsFactGraphValues(`${WITHOUT_FACT_PATHS_PREFIX}.${i18nKey}`);

      expect(foundFactGraphValues).toEqual(false);
    }
  );

  it.each(Object.entries(i18nTranslationsWithFactPaths[WITH_FACT_PATHS_PREFIX]))(
    `callback returns true if any translation key contains a fact path "${WITH_FACT_PATHS_PREFIX}.%s": %o`,
    (i18nKey, _i18nValue) => {
      const {
        result: { current: containsFactGraphValues },
      } = renderHook(useContainsFactGraphValues);

      const foundFactGraphValues = containsFactGraphValues(`${WITH_FACT_PATHS_PREFIX}.${i18nKey}`);

      expect(foundFactGraphValues).toEqual(true);
    }
  );
});
