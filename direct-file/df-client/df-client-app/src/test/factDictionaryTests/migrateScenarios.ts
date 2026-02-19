/* eslint-disable no-console */
import fs from 'fs';
import { format } from 'prettier';
import * as sfg from '@irs/js-factgraph-scala';
import { wrappedFacts } from '../../fact-dictionary/generated/wrappedFacts.js';
import { CURRENT_TAX_YEAR } from '../../constants/taxConstants.js';

/**
 * This file is used for migrating scenario JSON files.
 * WHY? When a change in the codebase requires that all jsons contain new facts
 * or updates to existing facts, this script can be used to update all of
 * the files at once.
 *
 * HOW? The script does not automatically detect anything or run automatically.
 * The eng who made the change to the codebase should change this script to update the facts as needed.
 * In most cases only the section between MIGRATION START and MIGRATION END will need to be updated.
 * Then the eng runs this script and commits the updated jsons.
 * Note that even though it is checked in, this script runs only once to update all the scenario jsons.
 *
 * To run this file from df-client-app:
 *
 * npm run migrate-scenarios
 */

// List the folders with scenarios to be migrated.
// (Note `./src/test/factDictionaryTests/backend-scenarios` is just a link to backend scenarios folder.)
const scenarioFolders = [
  `../../backend/src/test/resources/scenarios`,
  `../../backend/src/test/resources/scenarios-ero`,
];

// List any files to be excluded from the migration.
const filesToSkip = [`../../backend/src/test/resources/scenarios/ats-1-1099r-disabled.json`];

const jsonFiles = scenarioFolders.flatMap((folder: fs.PathLike) =>
  fs
    .readdirSync(folder)
    .filter((file) => file.endsWith(`.json`) && !filesToSkip.includes(folder + `/` + file))
    .map((file) => folder + `/` + file)
);

/** easy fact manipulation guide (pls add):
 * delete facts[pathName]
 */

jsonFiles.forEach((fileName) => {
  const jsonString = fs.readFileSync(fileName, `utf-8`);
  const factJson = JSON.parse(jsonString);

  const facts = factJson.facts ? factJson.facts : factJson;

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const factGraph = setupFactGraph(facts);

  // MIGRATION START

  // If there are 1099-Rs, set the feature flag
  if (facts[`/form1099Rs`]?.item.items.length > 0) {
    facts[`/is1099RFeatureFlagEnabled`] = {
      $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
      item: true,
    };
  }

  // MIGRATION END

  if (!factJson.facts) {
    const newJsonString = JSON.stringify({
      facts,
    });
    fs.writeFileSync(fileName, format(newJsonString, { parser: `json` }));
  } else {
    const newJsonString = JSON.stringify(factJson);
    fs.writeFileSync(fileName, format(newJsonString, { parser: `json` }));
  }
});

// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore: migrating from javascript to satisfy the build system.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function setupFactGraph(existingFacts) {
  // We use the scala fact graph directly instead of using our existing wrappers/interceptors
  // This prevents us from importing a bunch of react code and node struggling to import modules.scss files
  const meta = new sfg.DigestMetaWrapper(CURRENT_TAX_YEAR).toNative();

  // TODO: Write a type for the digest stuff
  const facts = wrappedFacts.map((fact) =>
    sfg.DigestNodeWrapperFactory.toNative(
      new sfg.DigestNodeWrapper(fact.path, fact.writable, fact.derived, fact.placeholder)
    )
  );

  const config = sfg.FactDictionaryConfig.create(meta, facts);

  const factDictionary = sfg.FactDictionaryFactory.fromConfig(config);

  const existingFactJsonString = JSON.stringify(existingFacts);
  const persister = sfg.JSPersister.create(existingFactJsonString);

  const factGraph = sfg.GraphFactory.apply(factDictionary, persister);
  return factGraph;
}
