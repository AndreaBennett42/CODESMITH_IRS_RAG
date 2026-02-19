import { facts } from './generated/facts.js';
import { readPdfFactPaths } from './generate-src/readPdfFactPaths.js';

const dictionaryPaths = facts.map((f) => f[`@path`]);
const factPathsUsedByPdf = readPdfFactPaths();
const missingPdfDependencies = factPathsUsedByPdf.filter((p) => !dictionaryPaths.includes(p));
if (missingPdfDependencies.length > 0) {
  throw new Error(
    `PDF dependencies include paths that do not exist in the fact dictionary:\n${missingPdfDependencies.join(`\n`)}`
  );
}
