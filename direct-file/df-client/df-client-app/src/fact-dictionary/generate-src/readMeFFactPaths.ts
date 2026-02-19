import fs from 'fs';
import { Path } from '../Path.js';
const MEF_PATHS_LOCATION = `./src/fact-dictionary/generate-src/xmlFactPaths`;

export function readMeFFactPaths() {
  const rawFileString = fs.readFileSync(MEF_PATHS_LOCATION, `utf-8`);
  const factPaths = rawFileString.split(`\n`).filter((s) => s.startsWith(`/`));
  return factPaths as Path[];
}
