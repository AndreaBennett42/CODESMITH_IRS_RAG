// eslint-disable-next-line import/default
import ExcelJS from 'exceljs';
import fs from 'fs';
import yaml from 'js-yaml';
import Locale, { RawLocale, countGrouped, normalizeCopy, translateKey } from './locale.js';

// eslint-disable-next-line import/no-named-as-default-member
const { Workbook } = ExcelJS;

const logger = console;

const languageDirectory = `src/locales`;
const esLocalePath = `${languageDirectory}/es.yaml`;
// nosemgrep: nodejs_scan.javascript-eval-rule-yaml_deserialize
const rawLocaleEs = yaml.load(fs.readFileSync(esLocalePath, `utf-8`));

const enLocalePath = `${languageDirectory}/en.yaml`;
// nosemgrep: nodejs_scan.javascript-eval-rule-yaml_deserialize
const rawLocaleEn = yaml.load(fs.readFileSync(enLocalePath, `utf-8`));

export function excelFix(copy: string | { richText: ExcelJS.RichText[] }) {
  if (typeof copy !== `string`) {
    // ExcelJS sometimes forgets to turn RichText back into text
    return copy.richText.map((val) => val.text).join(``);
  } else return copy;
}

function summaryCount(enLocale: Locale, langLocale: Locale) {
  const diff = langLocale.diff(enLocale);

  logger.log(`[info] English keys: ${diff.otherCount}`);
  logger.log(`[info] Spanish keys: ${diff.thisCount}`);
  logger.log(`[info] Missing keys: ${diff.removed.length}`);

  const grouped = countGrouped(diff.removed);
  logger.log(grouped);
  // logger.log(diff.removed.join(`\n`)); //Uncomment to see actual missing keys
}

function main() {
  const args = process.argv.slice(2);
  const usage = `Usage: import-locales <translation-column> <xls-path>`;
  // Check for 2 arguments
  if (args.length !== 2) {
    logger.error(`Not enough arguments. \n${usage}\n`);
    process.exit(1);
  }
  const [translationColumn, translationWorksheetPath] = args;

  // Check that file exists
  // nosemgrep: eslint.detect-non-literal-fs-filename
  if (!fs.existsSync(translationWorksheetPath)) {
    logger.error(`Error: The path '${translationWorksheetPath}' does not exist.\n${usage}\n`);
    process.exit(1);
  }
  logger.log(`[info] Processing: ${translationWorksheetPath}`);
  logger.log(`[info] Expecting translations in col: ${translationColumn}`);

  const enLocale = new Locale(rawLocaleEn as RawLocale);
  const flattenedEnLocale = enLocale.flatten();
  const langLocale = new Locale(rawLocaleEs as RawLocale);

  const extraKeys: string[] = [];

  const workbook = new Workbook();

  workbook.xlsx.readFile(translationWorksheetPath).then(() => {
    workbook.eachSheet((sheet) => {
      const columnIds = {
        // Initialize with defaults
        key: `A`,
        currentEn: `C`,
        updatedEn: `D`,
      };

      // Ignore any non-screener copy
      sheet.eachRow((row, i) => {
        if (i === 0) {
          // This is the header row, check if the columns have changed
          row.eachCell((cell) => {
            const text = excelFix(cell.text);
            switch (text) {
              case `key`:
                columnIds.key = text;
                break;
              case `English (current)`:
                columnIds.currentEn = text;
                break;
              case `English (updated copy)`:
                columnIds.updatedEn = text;
                break;
            }
          });
        }
        if (Array.isArray(row.values)) {
          // Get values from excel and do some pre-checks
          const key = normalizeCopy(row.getCell(columnIds.key).text);
          const updatedEs = normalizeCopy(excelFix(row.getCell(translationColumn).text));
          const lastKnownEn = normalizeCopy(
            excelFix(row.getCell(columnIds.currentEn).text || row.getCell(columnIds.updatedEn).text)
          );
          const currentEs = normalizeCopy((langLocale.get(key) ?? ``).toString());

          // Don't override current, translated copy with blanks
          // Some worksheets may be partial and intentionally only included updates for specific rows
          // NOTE: This will still write blank strings if there is no existing copy to override
          if (currentEs && !updatedEs) return;

          if (key.includes(`.`)) {
            if (!translateKey(key, enLocale, flattenedEnLocale, langLocale, updatedEs, lastKnownEn)) {
              extraKeys.push(key);
            }
          } else {
            if (key.includes(`Subcategory`) || key.includes(`Screen`)) {
              logger.log(`[info] Starting ${key}`);
            }
            if (!key) {
              logger.log(`[warn] No key found at row ${i}`);
            }
          }
        }
      });
    });

    summaryCount(enLocale, langLocale);
    fs.writeFileSync(esLocalePath, yaml.dump(langLocale.toJSON()));
    fs.writeFileSync(`./exported-locales/extra-keys.txt`, extraKeys.join(`\n`));
  });
}

main();
