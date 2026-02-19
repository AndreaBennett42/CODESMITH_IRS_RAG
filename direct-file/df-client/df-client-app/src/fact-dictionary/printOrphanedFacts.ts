import { readMeFFactPaths } from './generate-src/readMeFFactPaths.js';
import { wrappedFacts } from './generated/wrappedFacts.js';

/**
 * This script finds any writable facts that are not used in derived facts!
 *
 * Not all orphaned writable facts are indicative of a problem; some are used only in the flow.
 */
const factsUsedByMeF = readMeFFactPaths();
// eslint-disable-next-line eqeqeq
const writableFacts = wrappedFacts.filter((f) => f.writable != undefined);
// eslint-disable-next-line eqeqeq
const derivedFacts = wrappedFacts.filter((f) => f.derived != undefined);
const allDependencies = new Set([
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ...derivedFacts.flatMap((f) => findDependencyFacts((f as any as DerivedFactChild).derived, f as any as DerivedFact)),
  ...factsUsedByMeF,
]);
const orphans = writableFacts.filter((writable) => !allDependencies.has(writable.path));
// eslint-disable-next-line no-console
console.log(orphans.map((o) => o.path).join(`\n`));

interface DerivedFactChild {
  typeName: string;
  options: { path: string }; // this is not fully accurate but is accurate enough
  children: DerivedFactChild[];
  derived: DerivedFactChild;
}

interface DerivedFact {
  path: string;
  writable: never;
  derived: DerivedFactChild;
  placeholder: DerivedFactChild; // close enough
}

function findDependencyFacts(fact: DerivedFactChild, baseFact: DerivedFact) {
  const ret: string[] = [];
  // eslint-disable-next-line eqeqeq
  if (fact.typeName == `Dependency`) {
    let absolutePath = undefined;
    if (fact.options.path.startsWith(`.`)) {
      absolutePath = [...baseFact.path.split(`/`).slice(0, -1), fact.options.path.replace(`../`, ``)].join(`/`);
    }
    ret.push(absolutePath || fact.options.path);
  }
  for (const child of fact.children) {
    ret.push(...findDependencyFacts(child, baseFact));
  }
  return ret;
}
