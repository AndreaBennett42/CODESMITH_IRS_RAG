
/**
 * NOTE: In order to run this script, `"type": "module"` must be temporarily removed from package.json
 *  this temp change should *NOT* be committed.
 *
 * TODO: Make this work with `"type": "module"`
 */
import fs from 'fs';

import { Workbook, Column } from 'exceljs';
import yaml from 'js-yaml';

// TODO: Consolidate shared code into df-common
type LocaleGroup = Record<string, string | object> | object;

const languageDirectory = `src/locales`;
// nosemgrep: nodejs_scan.javascript-eval-rule-yaml_deserialize
const en = yaml.load(fs.readFileSync(`${languageDirectory}/en.yaml`, `utf-8`)) as LocaleGroup;
// nosemgrep: nodejs_scan.javascript-eval-rule-yaml_deserialize
const es = yaml.load(fs.readFileSync(`${languageDirectory}/es.yaml`, `utf-8`)) as LocaleGroup;

/**
 *  Recursively walks through a locale object and outputs all the i18n key
 *  strings needed to reach the leaf nodes (anything with a string value).
 */
function flattenLocale(locale: LocaleGroup): [string, string][] {
  return Object.entries(locale).flatMap(([key, valueOrGroup]) =>
    typeof valueOrGroup === `string`
      ? [[key, valueOrGroup]]
      : flattenLocale(valueOrGroup).map(([subKey, value]) => [`${key}.${subKey}`, value] as [string, string])
  );
}

const COPY_STYLE = { alignment: { wrapText: true } };
const HEADERS: Partial<Column>[] = [
  { header: `key`, key: `key`, width: 50 },
  { header: `English (current)`, key: `en`, width: 50, style: COPY_STYLE },
  {
    header: `English (updated copy)`,
    key: `enUpdated`,
    width: 50,
    style: COPY_STYLE,
  },
  { header: `English last updated`, key: `enLastUpdated`, width: 25 },
  { header: `Español (current)`, key: `es`, width: 50, style: COPY_STYLE },
  {
    header: `Español (updated copy)`,
    key: `esUpdated`,
    width: 50,
    style: COPY_STYLE,
  },
  { header: `Español last updated`, key: `esLastUpdated`, width: 25 },
];

export function exportLocalesAsWorkbook() {
  const flattenedEnLocale = flattenLocale(en);
  const flattenedEsLocale = new Map(flattenLocale(es));

  const workbook = new Workbook();
  const worksheet = workbook.addWorksheet();
  worksheet.columns = HEADERS;

  for (const [key, en] of flattenedEnLocale) {
    const es = flattenedEsLocale.get(key) ?? ``;
    worksheet.addRow({ key, en, es });
  }

  return workbook;
}

function main() {
  const workbook = exportLocalesAsWorkbook();
  const [isoDate] = new Date().toISOString().split(`t`);
  workbook.xlsx.writeFile(`exported-locales-${isoDate}.xlsx`);
}

main();
