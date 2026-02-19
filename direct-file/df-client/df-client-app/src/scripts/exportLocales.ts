// eslint-disable-next-line import/default
import ExcelJS from 'exceljs';
import { parseArgs } from 'node:util';
import fs from 'fs';
import yaml from 'js-yaml';

import flowNodes from '../flow/flow.js'; // TODO: figure out how use FlowConfig instead
import { CategoryNode, ScreenNode, SubSubcategoryDeclaration, SubSubcategoryNode } from '../flow/flowDeclarations.js';

import { findFlowNodesOfType } from '../flow/findFlowNodesOfType.js';
import { getExpectedScreenContentKeys, getExpectedSubSubCategoryContentKeys } from '../locales/flowLocaleHelpers.js';
import { BATCH_NAME, BATCH_NAMES } from '../flow/batches.js';
import { ReactElement } from 'react';
import { ScreenContentNode } from '../flow/ContentDeclarations.js';
import Locale, { getVariables, NESTED_TRANSLATION_REGEX, normalizeCopy, RawLocale } from './locale.js';

// eslint-disable-next-line import/no-named-as-default-member
const { Workbook } = ExcelJS;

// PLURALIZED_PREFIXES includes all the key prefixes that are plural at the top-level but singular in the provided path
const PLURALIZED_PREFIXES = [`heading`, `subheading`, `iconList`];
// IGNORED_NODES includes all node types that cause this script to error but do not map to a flow-defined i18nkey

const languageDirectory = `src/locales`;
const locales = {
  // nosemgrep: nodejs_scan.javascript-eval-rule-yaml_deserialize
  en: new Locale(yaml.load(fs.readFileSync(`${languageDirectory}/en.yaml`, `utf-8`)) as RawLocale),
  // nosemgrep: nodejs_scan.javascript-eval-rule-yaml_deserialize
  es: new Locale(yaml.load(fs.readFileSync(`${languageDirectory}/es.yaml`, `utf-8`)) as RawLocale),
} as const;

type AvailableLocale = keyof typeof locales;
type I18nKey = string;

type CopyRow = Record<AvailableLocale, string> & { key: I18nKey; batch?: BATCH_NAME };
type KeysByScreen = Map<string, CopyRow[]>;
type ScreensBySubcategory = Map<string, KeysByScreen>;

function getCopyFromLocale(key: I18nKey, locale: Locale, allowBlank = false): [I18nKey, string][] {
  const copy = locale.get(key);

  if (!(allowBlank || copy !== undefined)) throw new Error(`Unable to find matching key for ${key}`);

  if (typeof copy === `string`) {
    return [[key, normalizeCopy(copy)]];
  } else if (copy && typeof copy === `object`) {
    return Locale.flattenRecurse(copy).map(([subKey, value]) => [`${key}.${subKey}`, normalizeCopy(value)]);
  } else {
    return [[key, copy || ``]];
  }
}
function getCopyForAllLocales(key: I18nKey): CopyRow[] {
  const enCopy = getCopyFromLocale(key, locales.en);

  // Get the spanish copy (if available) of all related keys
  return enCopy.map(([key, en]) => ({ key, en, es: normalizeCopy((locales.es.get(key) ?? ``).toString()) }));
}

export function autoPrefix(i18nkey: string) {
  if (i18nkey.startsWith(`/`)) {
    // Key has an implicit prefix
    // Key is expected be similar to: /info/some-category/some-subcategory/some-screen
    const implicitPrefix = i18nkey.split(`/`)[1];
    const shouldPluralize = PLURALIZED_PREFIXES.includes(implicitPrefix);
    return `${implicitPrefix}${shouldPluralize ? `s` : ``}.${i18nkey}`;
  } else {
    return i18nkey;
  }
}

function keyMatchesFilter(key: string, filterKey: string) {
  if (filterKey.startsWith(`*`)) {
    return key.includes(filterKey.replace(`*`, ``));
  } else return key.startsWith(filterKey);
}

function keyMatchesAtLeastOnefilter(key: string, keysToExport: string[]) {
  return keysToExport.some((prefixOrKey) => keyMatchesFilter(key, prefixOrKey));
}

export function getScreenGroups({
  screensToExport,
  keysToExport,
  batchesToExport,
  missingOnly = false,
}: {
  screensToExport?: string[];
  keysToExport?: string[];
  batchesToExport?: BATCH_NAME[];
  missingOnly?: boolean;
}): Map<string, ScreensBySubcategory> {
  let keysSeen: I18nKey[] = [];
  let keysSeenAndOutput: I18nKey[] = [];
  const categories: CategoryNode[] = findFlowNodesOfType(flowNodes, `Category`);

  // Group screens by category and subcategory
  const screenGroups = new Map(
    categories
      .map(
        (categoryNode) =>
          [
            // Category group
            categoryNode.props.route as string,
            new Map(
              findFlowNodesOfType(categoryNode, `Subcategory`)
                .map((subcategoryNode) => {
                  const screens: ScreenNode[] = findFlowNodesOfType(subcategoryNode, `Screen`).filter(
                    (screen) => screensToExport?.includes(screen.props.route) ?? true
                  );
                  const categoryRoute = categoryNode.props.route;
                  const subCategoryRoute = subcategoryNode.props.route;
                  const parentRoute = `/flow/${categoryRoute}/${subCategoryRoute}`;

                  const keysByScreen: KeysByScreen = new Map(
                    screens
                      .map((screenNode) => {
                        // Coerce and narrow all children into an array
                        const childNodeOrNodes = screenNode.props.children;
                        const childNodes = Array.isArray(childNodeOrNodes) ? childNodeOrNodes : [childNodeOrNodes];

                        // Get all keys referenced in the fact config and child nodes
                        const baseScreenKeys = childNodes
                          .flatMap((childNode) => {
                            const { requiredKeys, optionalKeys } = getExpectedScreenContentKeys(childNode, parentRoute);

                            let keysWithBatchInformation = [
                              ...requiredKeys,
                              ...optionalKeys.filter((key) => locales.en.has(key)),
                            ].map((key) => ({
                              key,
                              batches: childNode.props.batches,
                            }));

                            // Add in all child content
                            const childNodeOrNodes =
                              ((childNode as ReactElement).props.children as ScreenContentNode[]) ?? [];

                            const childNodes = Array.isArray(childNodeOrNodes) ? childNodeOrNodes : [childNodeOrNodes];

                            if (childNodes.length > 0) {
                              for (const subChildNode of childNodes) {
                                const subContent = getExpectedScreenContentKeys(subChildNode, parentRoute);
                                keysWithBatchInformation = keysWithBatchInformation.concat(
                                  [
                                    ...subContent.requiredKeys,
                                    ...subContent.optionalKeys.filter((key) => locales.en.has(key)),
                                  ].map((key) => ({
                                    key,
                                    batches: subChildNode.props.batches,
                                  }))
                                );
                              }
                            }

                            return keysWithBatchInformation;
                          })
                          .filter(({ key }) => !keysSeenAndOutput.includes(key));

                        // Remove any duplicates found within the same screen and merge batch lists
                        let screenKeys = Array.from(
                          baseScreenKeys
                            .reduce((accumulator, { key, batches }) => {
                              if (!accumulator.has(key)) {
                                accumulator.set(key, new Set(batches));
                              } else if (batches && batches.length > 0) {
                                // Merge sets
                                const prevBatches = accumulator.get(key) as Set<BATCH_NAME>;

                                accumulator.set(key, new Set([...batches, ...prevBatches]));
                              }

                              return accumulator;
                            }, new Map<string, Set<BATCH_NAME>>())
                            .entries()
                        ).flatMap(([key, batches]) =>
                          getCopyForAllLocales(key).map((copy) => ({ ...copy, batches: Array.from(batches) }))
                        );

                        // Mark the keys found in this screen as "seen"
                        keysSeen = keysSeen.concat(screenKeys.map(({ key }) => key));

                        if (batchesToExport) {
                          screenKeys = screenKeys.filter(({ batches }) =>
                            batches?.some((batch) => batchesToExport.includes(batch))
                          );
                        }

                        keysSeenAndOutput = [...keysSeenAndOutput, ...screenKeys.map(({ key }) => key)];

                        return [
                          screenNode.props.route,
                          screenKeys.filter(
                            ({ key, en, es }) =>
                              // If filtering by key, ensure it matches at least one provided filter
                              (keysToExport === undefined || keyMatchesAtLeastOnefilter(key, keysToExport)) &&
                              // If requested, only include content which has English copy, but not Spanish copy
                              (!missingOnly || (en && !es))
                          ),
                        ] as [string, CopyRow[]];
                      })
                      .filter((screenKeys) => screenKeys[1].length > 0)
                  );

                  const subSubcategories: SubSubcategoryNode[] = findFlowNodesOfType(
                    subcategoryNode,
                    `SubSubcategory`
                  ).filter((subSubCategory) => !(subSubCategory.props as SubSubcategoryDeclaration).hidden);

                  if (subSubcategories.length > 0) {
                    const subSubcategoryKeys = subSubcategories.reduce((keys, subSubcategory) => {
                      const expectedKeys = getExpectedSubSubCategoryContentKeys(subSubcategory, parentRoute);

                      const flattenedKeys = [
                        ...expectedKeys.requiredKeys,
                        ...expectedKeys.optionalKeys.filter((key) => locales.en.has(key)),
                      ].flatMap(getCopyForAllLocales);

                      keysSeen = keysSeen.concat(flattenedKeys.map(({ key }) => key));

                      // Don't export subSubcategory keys unless one of the following is true
                      // * we're exporting at least one screen inside of it
                      // * we are not filtering by screen or batch
                      if (
                        !(
                          !(screensToExport || batchesToExport) ||
                          findFlowNodesOfType(subSubcategory, `Screen`).some((screen) =>
                            keysByScreen.has(screen.props.route)
                          )
                        )
                      ) {
                        return keys;
                      }

                      return Array.from(new Set([...keys, ...flattenedKeys])).filter(
                        ({ key }) => keysToExport === undefined || keyMatchesAtLeastOnefilter(key, keysToExport)
                      );
                    }, [] as CopyRow[]);

                    keysSeenAndOutput = [...keysSeenAndOutput, ...subSubcategoryKeys.map(({ key }) => key)];

                    const subSubcategoryCopy = subSubcategoryKeys.filter(
                      ({ key, en, es }) =>
                        // If filtering by key, ensure it matches at least one provided filter
                        (keysToExport === undefined || keyMatchesAtLeastOnefilter(key, keysToExport)) &&
                        // If requested, only include content which has English copy, but not Spanish copy
                        (!missingOnly || (en && !es))
                    );

                    if (subSubcategoryCopy.length > 0) {
                      keysByScreen.set(`SubSubCategories`, subSubcategoryCopy);
                    }
                  }

                  return [
                    // Subcategory group
                    subcategoryNode.props.route as string,
                    keysByScreen,
                  ] as const;
                })
                .filter(([_, keysByScreen]) => keysByScreen.size > 0)
            ),
          ] as const
      )
      .filter(([_, subcategoriesByCategory]) => subcategoriesByCategory.size > 0)
  );

  // Group any keys in the en locale not found while walking screens
  const unseenCopy: CopyRow[] = locales.en
    .flatten()
    .filter(
      ([key]) =>
        // eslint-disable-next-line eqeqeq
        !keysSeen.some((seenKey) => key == seenKey) &&
        (keysToExport === undefined || keyMatchesAtLeastOnefilter(key, keysToExport))
    )
    .flatMap(([key]) => getCopyForAllLocales(key))
    .filter(({ en, es }) => !missingOnly || (en && !es));

  if ((keysToExport || !(screensToExport || batchesToExport)) && unseenCopy.length > 0) {
    const unseenCategory: ScreensBySubcategory = unseenCopy.reduce((result, copy) => {
      const keyParts = copy.key.split(`.`);

      const subcategoryKey = keyParts[0];
      const kindaScreenRoute = keyParts[1];

      if (!result.has(subcategoryKey)) {
        result.set(subcategoryKey, new Map());
      }
      const subcategory: KeysByScreen = result.get(subcategoryKey) as KeysByScreen;

      if (!subcategory.has(kindaScreenRoute)) {
        subcategory.set(kindaScreenRoute, []);
      }
      const kindaScreen = subcategory.get(kindaScreenRoute) as CopyRow[];

      kindaScreen.push(copy);

      return result;
    }, new Map<string, KeysByScreen>());
    screenGroups.set(`Unseen Copy`, unseenCategory);
  }

  return screenGroups;
}

const COPY_STYLE = { alignment: { wrapText: true } };
const SUBCATEGORY_COLOR = `FFD0D0D0`;
const SCREEN_COLOR = `D8E4BC`;
const UNTRANSLATED_COLOR = `FFD700`;

function setRowBackground(row: ExcelJS.Row, color: string) {
  row.eachCell({ includeEmpty: true }, (cell) => {
    // Set background color on the cell
    cell.style.fill = {
      type: `pattern`,
      pattern: `solid`,
      fgColor: { argb: color },
    };

    // If cell is completely empty, the row styles aren't retained
    if (!cell.value) cell.value = ``;
  });
}

function stringContainsTranslatableContent(copy: string) {
  const variables = getVariables(copy);

  const strippedCopy = copy
    // Remove any nested translations
    .replaceAll(NESTED_TRANSLATION_REGEX, ``)
    // Remove any variable references
    .replaceAll(new RegExp(`(${variables.join(`|`)})`, `g`), ``);

  // Assume that any remaining letters are part of some translatable words
  return strippedCopy.match(/[A-z]+/g);
}

function getTranslationSuggestion({ key, en }: CopyRow) {
  const reverseMap = locales.en.getReverseMap();

  const matchingKeys = reverseMap.get(en) ?? new Set();
  matchingKeys.delete(key); // Matching with itself would be silly

  if (matchingKeys.size > 0) {
    const candidateTranslations = new Set(
      Array.from(matchingKeys)
        .filter((key) => locales.es.has(key))
        .map((key) => normalizeCopy(locales.es.get(key).toString()))
    );

    return {
      matchingKeys: Array.from(matchingKeys),
      candidateTranslations: Array.from(candidateTranslations).filter((translation) => translation.length > 0),
    };
  } else {
    return { matchingKeys: [], candidateTranslations: [] };
  }
}

const HEADERS: Partial<ExcelJS.Column>[] = [
  { header: `key`, key: `key`, width: 50 },
  { header: `batches`, key: `batches`, width: 20 },
  { header: `English (current)`, key: `en`, width: 50, style: COPY_STYLE },
  { header: `English (updated copy)`, key: `enUpdated`, width: 50, style: COPY_STYLE },
  { header: `English last updated`, key: `enLastUpdated`, width: 25 },
  { header: `Español (current)`, key: `es`, width: 50, style: COPY_STYLE },
  { header: `Español (updated copy)`, key: `esUpdated`, width: 50, style: COPY_STYLE },
  { header: `Español last updated`, key: `esLastUpdated`, width: 25 },
];
export function exportLocalesAsWorkbook({
  highlightMissing = false,
  suggestAlternateTranslations = false,
  ...exportOptions
}: {
  screensToExport?: string[];
  keysToExport?: string[];
  missingOnly?: boolean;
  batchesToExport?: BATCH_NAME[];
  highlightMissing?: boolean;
  suggestAlternateTranslations?: boolean;
}) {
  const screenGroups = getScreenGroups(exportOptions);

  const workbook = new Workbook();

  for (const [categoryRoute, screensBySubcategory] of screenGroups.entries()) {
    const worksheet = workbook.addWorksheet(categoryRoute, {
      /* TODO: header/footer? */
    });
    worksheet.columns = HEADERS;

    for (const [subcategoryRoute, keysByScreen] of screensBySubcategory.entries()) {
      const subcategoryRow = worksheet.addRow({ key: `Subcategory: ${subcategoryRoute}`, esLastUpdated: ` ` });
      setRowBackground(subcategoryRow, SUBCATEGORY_COLOR);

      for (const [screenRoute, screenKeys] of keysByScreen.entries()) {
        const screenRow = worksheet.addRow({ key: `Screen: ${screenRoute}`, esLastUpdated: ` ` });
        setRowBackground(screenRow, SCREEN_COLOR);

        for (const copy of screenKeys) {
          const copyRow = worksheet.addRow(copy);

          const esIsMissing = copy.es.length === 0;
          const esCell = copyRow.getCell(`es`);
          if (!stringContainsTranslatableContent(copy.en)) {
            if (esIsMissing) {
              // There's nothing that really needs translating, so let's just prefill with the english
              esCell.value = copy.en;

              esCell.note = `Prefilled with English copy, as English doesn't seem to contain any translatable content`;
            } else if (copy.en !== copy.es) {
              esCell.note =
                `English doesn't seem to contain any translatable content, ` +
                `but the existing English and Spanish have diverged`;
            }
          } else if (suggestAlternateTranslations) {
            const { matchingKeys, candidateTranslations } = getTranslationSuggestion(copy);

            if (matchingKeys.length > 0) {
              if (candidateTranslations.length > 1) {
                esCell.note = `Found multiple candidate translation suggestions:\n${Array.from(
                  candidateTranslations
                ).join(`\n\n----\n\n`)}`;
              } else if (candidateTranslations.length === 1) {
                const translationSuggestion = normalizeCopy(candidateTranslations[0]);

                if (!copy.es) {
                  esCell.value = translationSuggestion;
                  esCell.note = `Suggested translation copied from [\n${Array.from(matchingKeys).join(`,\n`)}\n]`;
                } else if (normalizeCopy(copy.es) !== translationSuggestion) {
                  esCell.note =
                    `Found duplicate English content with an alternate translation:\n` +
                    `Source key(s): ${Array.from(matchingKeys).join(`,\n`)}\n` +
                    `Alternate translation: ${translationSuggestion}`;
                }
              } else {
                esCell.note = `Found duplicate english content, but no Spanish translation has been found.${
                  `` // forcing a line break
                } Matching keys: [\n${Array.from(matchingKeys).join(`,\n`)}]`;
              }
            }
          }

          if (esCell.note) {
            setRowBackground(copyRow, UNTRANSLATED_COLOR);
          }

          if (highlightMissing && esIsMissing) {
            setRowBackground(copyRow, UNTRANSLATED_COLOR);
          }
        }
      }
    }
  }

  return workbook;
}

function loadListFromFilePath(screenListPath: string) {
  // nosemgrep: eslint.detect-non-literal-fs-filename
  return fs
    .readFileSync(screenListPath, `utf-8`)
    .split(/\s+/g)
    .map(normalizeCopy) // Make sure any weird whitespace characters are properly stripped
    .filter((path) => path.length > 0);
}

function parseOrLoadBatchList(batchListOrFilePath: string) {
  const maybeBatchList = batchListOrFilePath.split(/\s+/g);

  if (maybeBatchList.every((maybeBatch) => BATCH_NAMES.includes(maybeBatch as BATCH_NAME))) {
    return maybeBatchList;
  } else {
    return loadListFromFilePath(batchListOrFilePath);
  }
}

function main() {
  const {
    values: { screenListPath, keyListPath, batchesPath, ...rest },
  } = parseArgs({
    options: {
      screenListPath: {
        type: `string`,
        short: `s`,
      },
      keyListPath: {
        type: `string`,
        short: `k`,
      },
      missingOnly: {
        type: `boolean`,
        short: `m`,
      },
      highlightMissing: {
        type: `boolean`,
        short: `h`,
      },
      batchesPath: {
        type: `string`,
        short: `b`,
      },
      suggestAlternateTranslations: {
        type: `boolean`,
        short: `d`,
      },
    },
    allowPositionals: true,
  });

  const workbook = exportLocalesAsWorkbook({
    screensToExport: screenListPath ? loadListFromFilePath(screenListPath) : undefined,
    keysToExport: keyListPath ? loadListFromFilePath(keyListPath) : undefined,
    batchesToExport: batchesPath ? (parseOrLoadBatchList(batchesPath) as BATCH_NAME[]) : undefined,
    ...rest,
  });
  const [isoDate] = new Date().toISOString().split(`t`);
  workbook.xlsx.writeFile(`exported-locales/${isoDate}.xlsx`);
}

main();
