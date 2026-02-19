import { Workbook } from 'exceljs';
import fs from 'fs';
import yaml from 'js-yaml';
const logger = console;

type RawLocale = Record<string, string | object>;

const languageDirectory = `src/locales`;
const esLocalePath = `${languageDirectory}/es.yaml`;
// nosemgrep: nodejs_scan.javascript-eval-rule-yaml_deserialize
const rawLocaleEs = yaml.load(fs.readFileSync(esLocalePath, `utf-8`));

class Locale {
  constructor(protected rawLocale: RawLocale) {}

  get(i18nKey: string) {
    return i18nKey
      .split(`.`)
      .reduce(
        (localeGroup, keyPart) => (localeGroup as Record<string, string | object>)?.[keyPart],
        this.rawLocale as string | object
      );
  }

  set(i18nKey: string, value: string) {
    const keyPath = i18nKey.split(`.`);

    // Follow the key path, creating subjects along the way as needed
    keyPath.reduce((localeGroup, keyPart, index) => {
      if (index === keyPath.length - 1) {
        localeGroup[keyPart] = value;
        return {}; // Return an object to appease the typing gods
      } else {
        if (typeof localeGroup[keyPart] !== `object`) {
          // TODO: Identify when arrays should be initialized instead
          localeGroup[keyPart] = {};
        }

        // console.log(`Continuing to iterate`);
        return localeGroup[keyPart] as RawLocale;
      }
    }, this.rawLocale);
  }

  toJSON() {
    // TODO: See if we can normalize any numerically keyed objects into arrays
    return this.rawLocale;
  }
}

function main() {
  // TODO: Add actually arg handling and help options
  const translationWorksheetPath = process.argv[2];

  if (!translationWorksheetPath) {
    throw new Error(`You must specify a path to the worksheet you want to import`);
  }

  const locale = new Locale(rawLocaleEs as RawLocale);

  const workbook = new Workbook();
  workbook.xlsx.readFile(translationWorksheetPath).then(() => {
    workbook.eachSheet((sheet) => {
      // Ignore any non-screener copy
      if (sheet.name !== `Screener`) return;

      sheet.eachRow((row, i) => {
        if (Array.isArray(row.values)) {
          const key = row.getCell(`A`).toString();
          const updatedEs = row.getCell(`H`).toString();
          if (key?.includes(`.`)) {
            locale.set(key, updatedEs);
          } else {
            if (!key) {
              logger.log(`Row ${i} needs an key`);
            }
          }
        }
      });
    });

    fs.writeFileSync(esLocalePath, yaml.dump(locale.toJSON()));
  });
}

main();
