/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable no-var */
import fs from 'fs';
import { Path as Path2 } from '../../flow/Path.js';
import { format } from 'prettier';
import { InterceptingFactGraph } from '../../factgraph/InterceptingFactGraph.js';
import { Path } from '../../fact-dictionary/Path.js';
import { RawFact } from '../../fact-dictionary/FactTypes.js';
// These have to be added for the intercepting fact graph to be happy
declare global {
  var debugFactGraph: any;
  var debugFacts: any;
  var debugFactGraphMeta: any;
  var debugScalaFactGraphLib: any;
  var rawFacts: RawFact[];
  var loadFactGraph: (json: string) => void;
  var saveFactGraphToLocalStorageKey: (keyId: string, force?: boolean) => void;
  var loadFactGraphFromLocalStorageKey: (keyId: string) => void;
}

const SCENARIO_FOLDER = `./src/test/factDictionaryTests/backend-scenarios`;
const files = fs.readdirSync(SCENARIO_FOLDER);
const jsons = files.filter((f) => f.endsWith(`.json`));

const TESTED_FACTS: Path[] = [
  `/isFilingStatusHOH`,
  `/isFilingStatusMFJ`,
  `/isFilingStatusMFS`,
  `/isFilingStatusMFS`,
  `/isFilingStatusSingle`,
  `/isReceivingActc`,
  `/isReceivingCtc`,
  `/isReceivingOdc`,
  `/isReceivingEitc`,
  `/primaryFiler/canBeClaimed`,
  `/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`,
  `/secondaryFiler/canBeClaimed`,
  `/isMFJDependent`,
];
const counter: { [key: string]: number } = {};

jsons.forEach((json) => {
  const fileName = SCENARIO_FOLDER + `/` + json;
  const jsonString = fs.readFileSync(fileName, `utf-8`);
  const factJson = JSON.parse(jsonString);

  const facts = factJson.facts;

  const factGraph = new InterceptingFactGraph(facts);
  TESTED_FACTS.forEach((fact) => {
    const isTrue =
      factGraph.get(Path2.concretePath(fact, null)).complete &&
      // eslint-disable-next-line eqeqeq
      factGraph.get(Path2.concretePath(fact, null)).get == true;
    if (counter[fact] === undefined) {
      counter[fact] = 0;
    }
    if (isTrue) {
      counter[fact]++;
    }
  });
});

process.stdout.write(`${jsons.length} fact graphs tested\n`);
process.stdout.write(format(JSON.stringify(counter), { parser: `json` }));
