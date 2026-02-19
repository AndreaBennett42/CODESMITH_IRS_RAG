/**
 * Temporary script to make it easy to merge in-flight changes to the json files into yaml
 */
import fs from 'fs';
import yaml from 'js-yaml';
import { YamlSettings } from '../locales/yaml-settings.js';

/**
 * Path to locales folder, relative to the df-client-app project root
 */
const LOCALE_ROOT = `${__dirname}/../locales`;

function handleWriteFileError(targetName: string) {
  return (error: NodeJS.ErrnoException | null) => {
    if (error) {
      return null;
    }
  };
}

function convertLocales() {
  // nosemgrep: eslint.detect-non-literal-fs-filename
  const files = fs.readdirSync(LOCALE_ROOT, { withFileTypes: true });

  for (const file of files.filter((file) => file.name.endsWith(`.json`))) {
    const targetName = file.name.replace(`json`, `yaml`);

    import(`${LOCALE_ROOT}/${file.name}`).then((jsonLocale) =>
      // nosemgrep: eslint.detect-non-literal-fs-filename
      fs.writeFile(
        `${LOCALE_ROOT}/${targetName}`,
        yaml.dump(jsonLocale, YamlSettings),
        handleWriteFileError(targetName)
      )
    );
  }
}

convertLocales();
