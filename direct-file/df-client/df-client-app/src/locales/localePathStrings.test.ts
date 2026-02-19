import yaml from 'js-yaml';
import fs from 'fs';
import { pathReferencePattern } from '../hooks/useTranslationContextFromFacts.js';
import { paths } from '../fact-dictionary/generated/paths.js';
import { Path } from '../fact-dictionary/Path.js';
import Locale, { getVariables, NESTED_TRANSLATION_REGEX, RawLocale } from '../scripts/locale.js';
const languageDirectory = `src/locales`;
import { expect } from 'vitest';

const allLangs = [
  { lang: `English`, file: `en.yaml` },
  { lang: `Spanish`, file: `es.yaml` },
];

describe(`Locale strings links and PII`, () => {
  allLangs.forEach(({ lang, file }) => {
    describe(`in ${lang}`, () => {
      const langYaml = yaml.load(fs.readFileSync(`${languageDirectory}/${file}`, `utf-8`)) as object;
      let translationStrings: string[] = [];
      it(`Every string in ${file} is populated correctly`, () => {
        translationStrings = getLeafStrings(langYaml);
      });
      it(`Every path in ${file} appears is a valid path`, () => {
        const amendedPaths = [
          ...paths,
          // Due to some aliasing, certain paths might not appear in paths.ts, and
          // this is a short hack to make this test pass. One day we might make paths.ts better
          // and then this can be removed.
          `/form1099Gs/*/filer/fullName`,
          `/interestReports/*/filer/fullName`,
          `/socialSecurityReports/*/filer/fullName`,
          `/form1099Gs/*/filer/fullName`,
          `/primaryFiler/dateOfBirth/day`,
          `/primaryFiler/dateOfBirth/month`,
          `/primaryFiler/dateOfBirth/year`,
          `/secondaryFiler/dateOfBirth/day`,
          `/secondaryFiler/dateOfBirth/month`,
          `/secondaryFiler/dateOfBirth/year`,
          `/bankAccount/accountType`,
          `/bankAccount/routingNumber`,
          `/bankAccount/accountNumber`,
          `/oneAndOnlyW2/employerName`,
        ];

        // These lines are copy-pasted from `useTranslateWithFacts.ts`
        const rawPaths = translationStrings.flatMap((t) => [...t.matchAll(pathReferencePattern)].map((mg) => mg[1]));
        for (const path of rawPaths) {
          expect(amendedPaths).toContain(path);
        }
      });

      it(`getLinkText matches all modal links`, () => {
        const testStr = `<LinkModal12>Hello</LinkModal12> more <LinkPub501>Moar</LinkPub501>`;
        const matches = getLinkTexts(testStr);
        expect(matches.length).toBe(2);
        expect(matches[0]).toBe(`Hello`);
        expect(matches[1]).toBe(`Moar`);
      });

      it(`Facts containing PII do not appear in modal link, which would expose PII to analytics`, () => {
        // If this test fails, but your link text does not contain user information, you can add it to ALLOWED_FACTS
        const ALLOWED_FACTS: Set<Path> = new Set([`/taxYear`, `/nextTaxYear`, `/defaultTaxDay`, `/lastTaxYear`]);

        const stringsWithLinks = translationStrings.filter(
          (str) => str.includes(`<LinkModal`) || str.includes(`<InternalLink>`)
        );
        const linkTexts = stringsWithLinks.map((str) => {
          return { str, linkText: getLinkTexts(str) };
        });
        const errors: { str: string; path: Path }[] = [];
        for (const lt of linkTexts) {
          for (const linkText of lt.linkText) {
            const pathsInLinkText = [...linkText.matchAll(pathReferencePattern)].map((mg) => mg[1]) as Path[];
            for (const path of pathsInLinkText) {
              if (!ALLOWED_FACTS.has(path)) {
                errors.push({ str: lt.str, path: path });
              }
            }
          }
        }

        expect(errors).toStrictEqual([]);
      });
    });
  });
});

// Throughout the app we should be using ’ over '.
// The exceptions are user entered values where we often accept ' over ’.
describe(`should use right single quote (’)`, () => {
  const exceptionList = [
    `fields.generics.limitingString.errorMessages.InvalidEmployerNameLine1Characters`,
    `fields.generics.limitingString.errorMessages.InvalidEmployerNameLine2Characters`,
    `fields./cdccCareProviders/*/writableOrganizationName.errorMessages.InvalidEmployerNameLine1Characters`,
  ];
  allLangs.forEach(({ file }) => {
    it(`test in ${file}`, () => {
      const langYaml = yaml.load(fs.readFileSync(`${languageDirectory}/${file}`, `utf-8`)) as object;
      const flatObj = flatObject(langYaml);
      for (const key of Object.keys(flatObj)) {
        if (!exceptionList.includes(key)) {
          expect(
            flatObj[key].indexOf(`'`) === -1,
            `error in ${key}, found apostrophe (') instead of right single quote (’)`
          ).toBe(true);
        } else {
          expect(
            flatObj[key].indexOf(`'`) !== -1,
            `error in ${key} in exception list, found right single quote (’) instead of apostrophe (')`
          ).toBe(true);
        }
      }
    });
  });
});

describe.each(allLangs)(`Heading content in $lang`, ({ file }) => {
  // TODO: Refactor other tests to use `Locale` so we only need to load these once
  const langYaml = yaml.load(fs.readFileSync(`${languageDirectory}/${file}`, `utf-8`)) as RawLocale;
  const locale = new Locale(langYaml);

  const headings = locale.flatten().filter(([key]) => key.startsWith(`headings.`));

  const disallowedTags = [`p`, `ul`, `ol`, `li`, `div`, `h2`, `h3`, `h4`, `h5`, `h6`];

  it(`should not have any invalid tags inside`, () => {
    const invalidHeadingKeys = headings
      .filter(([_, copy]) => disallowedTags.some((tag) => copy.includes(`<${tag}>`)))
      .map(([key]) => key);

    expect(invalidHeadingKeys).toEqual([]);
  });
});

function flatObject(lang: object) {
  const flatLocale: [string, string][] = new Locale(lang as RawLocale).flatten();
  const flatObject = flatLocale.reduce<{ [key: string]: string }>((obj, [key, value]) => {
    obj[key] = value;
    return obj;
  }, {});
  return flatObject;
}

function resolveNestedTranslations(
  key: string,
  flattenedLocale: Record<string, string>,
  lang?: string,
  ancestorKeys: string[] = []
): string {
  const content = flattenedLocale[key];
  if (!content) return content;

  const nestedKeys = Array.from(content.matchAll(NESTED_TRANSLATION_REGEX)).map(([_, path]) => path);

  if (nestedKeys.length === 0) return content;

  return nestedKeys.reduce((result, i18nKey) => {
    if (ancestorKeys.includes(i18nKey)) {
      throw new Error(
        `${lang ? `${lang}: ` : ``}Discovered circular dependency in ${key}; referenced through ${ancestorKeys.join(
          ` -> `
        )}`
      );
    }
    const referencedContent = resolveNestedTranslations(i18nKey, flattenedLocale, lang, [...ancestorKeys, key]);
    if (!referencedContent) {
      throw new Error(
        `${lang ? `${lang}: ` : ``}Key '${key}' included nested translation to non-existent content (${i18nKey})`
      );
    }
    return result.replace(`$t(${i18nKey})`, referencedContent);
  }, content);
}

// Skipping this test because the translation files are now managed by Weblate
describe.skip(`Locale string variable matches`, () => {
  const enYaml = yaml.load(fs.readFileSync(`${languageDirectory}/en.yaml`, `utf-8`)) as object;
  const enFlatObject = flatObject(enYaml);
  const langs = [{ lang: `Spanish`, file: `es.yaml` }];
  langs.forEach(({ lang, file }) => {
    it(`in ${lang}`, () => {
      let errorCount = 0;
      const errors: object[] = [];
      const langYaml = yaml.load(fs.readFileSync(`${languageDirectory}/${file}`, `utf-8`)) as object;
      const langFlatObject = flatObject(langYaml);

      // Loop through all keys in English
      Object.keys(enFlatObject).forEach((key) => {
        // Find content in English and Spanish
        const enContent = resolveNestedTranslations(key, enFlatObject, `English`);
        const langContent = resolveNestedTranslations(key, langFlatObject);

        if (langContent && enContent) {
          const varsInEnglish = getVariables(enContent);
          const varsInLang = getVariables(langContent || ``);

          varsInEnglish.forEach((match) => {
            if (!varsInLang.includes(match)) {
              errorCount++;
              errors.push({
                key,
                [`inEnglishButNotIn${lang}`]: match,
                EnglishContent: enContent,
                [`${lang}Content`]: langContent,
              });
              // in case of errors, uncomment here and below to generate a CSV that you can pass to a translator
              // console.log(
              //   `"${key}","${match.replace(/"/g, `""`)}","${enContent.replace(/"/g, `""`)}","${langContent.replace(
              //     /"/g,
              //     `""`
              //   )}"`
              // );
            }
          });

          varsInLang.forEach((match) => {
            if (!varsInEnglish.includes(match)) {
              errorCount++;
              errors.push({
                key,
                [`in${lang}ButNotInEnglish`]: match,
                EnglishContent: enContent,
                [`${lang}Content`]: langContent,
              });
              // in case of errors, uncomment here and above to generate a CSV that you can pass to a translator
              // console.log(
              //   `"${key}","${match.replace(/"/g, `""`)}","${enContent.replace(/"/g, `""`)}","${langContent.replace(
              //     /"/g,
              //     `""`
              //   )}"`
              // );
            }
          });
        }
      });
      if (errorCount > 0) {
        // eslint-disable-next-line no-console
        console.log(`Errors: `, errors);
        expect.fail(`Test failed due to variable mismatches between language files. See the errors in the console.`);
      }
    });
  });
});

const linkRegex = /<Link[a-zA-Z0-9]*>(.*?)<\/Link[a-zA-Z0-9]*>/g;
const internalLinkRegex = /<InternalLink>(.*?)<\/InternalLink>/g;

function getLinkTexts(translationString: string) {
  const internalLinkMatches = [...translationString.matchAll(internalLinkRegex)].map((match) => match[1]);
  const matches = [...translationString.matchAll(linkRegex)].map((match) => match[1]);
  return [...matches, ...internalLinkMatches];
}

function getLeafStrings(obj: object, parentKey?: string): string[] {
  const ret: string[] = [];

  try {
    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === `string`) {
        ret.push(value);
      } else {
        const newKey = parentKey ? [parentKey, key].join(`.`) : key;
        ret.push(...getLeafStrings(value, newKey));
      }
    }
  } catch (e) {
    if (e instanceof TypeError) {
      // Capture the source of the error
      throw new Error(`Something is wrong with this entry in the locale file \n${parentKey}`);
    } else {
      throw e;
    }
  }
  return ret;
}
