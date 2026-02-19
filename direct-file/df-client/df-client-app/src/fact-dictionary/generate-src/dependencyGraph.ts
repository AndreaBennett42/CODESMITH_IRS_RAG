import {
  addressFields,
  bankAccountFields,
  dependentCollectionAliases,
  individualFilerAliases,
  filerCollectionAliases,
  individualDependentAliases,
  form1099MiscAliases,
  form1099RAliases,
  w2Aliases,
  individualW2Aliases,
  hsaDistributionsAliases,
  interestReportsAliases,
  form1099GAliases,
} from '../aliases.js';
import { Path } from '../Path.js';
import { wrappedFacts } from '../generated/wrappedFacts.js';
import { CompNodeConfigDigestWrapper } from '@irs/js-factgraph-scala';
import { assertNever } from 'assert-never';
import { tinSubfields } from './subfields.js';

export interface Dependency {
  module?: string;
  path: Path;
}

export function generateDependencyGraph() {
  const map = new Map<Path, Dependency[]>();
  for (const fact of wrappedFacts) {
    if (fact.derived) {
      map.set(fact.path, getDependencies(fact.derived));
    } else if (fact.writable) {
      map.set(fact.path, []);
    } else {
      assertNever(fact);
    }
  }
  for (const [factPath, dependents] of map.entries()) {
    const first2 = parsePathFirst2(factPath);
    const fullDependents = dependents.map((pathToDependency) => {
      let genericPath: Path | undefined = pathToDependency.path;
      if (first2) {
        genericPath = unmapFactAlias(genericPath.replace(`..`, first2)) as Path;
      }
      return { path: genericPath, module: pathToDependency.module };
    }) as (Dependency | undefined)[];
    map.set(
      factPath,
      // eslint-disable-next-line eqeqeq
      fullDependents.filter((d): d is Dependency => d?.path != undefined)
    );
  }
  return map;
}

const first2Regex = new RegExp(`((/.*/*/)(.*))/`);

export function parsePathFirst2(path: Path) {
  const res = path.match(first2Regex);
  return res?.[1];
}

function getDependencies(node: CompNodeConfigDigestWrapper): Dependency[] {
  if (node.typeName === `Dependency`) {
    return [{ path: unmapFactAlias(node.options.path), module: node.options.module }];
    // We should consider generating synthetic nodes for Any/All conditions, so we can better
    // follow and debug paths
    // eslint-disable-next-line eqeqeq
  } else if (node.typeName === `Filter` || node.typeName == `Find`) {
    // These dependencies will have incomplete paths like `eligibleDependent`
    // upon which we need to prefix our collection to make it look like `familyAndHousehold/*/eligibleDependent`
    const dependencies = node.children.flatMap((child) => getDependencies(child));
    const collection = node.options.path;
    return dependencies.map((d) => {
      return {
        ...d,
        path: unmapFactAlias(d.path.startsWith(`/`) ? d.path : `${collection}/*/${d.path}`),
      };
    });
  } else if (node.children.length > 0) {
    return Array.from(new Set(node.children.flatMap(getDependencies)));
  } else return [];
}

// HACKHACK this sort of unaliasing is really useful for delivering the product we need now
// but is also very fragile. It could be improved a lot through better integration with the scala
// code. It is liable to break if additional path methods (like `isSSN`) get added onto facts,
// but the bright side is that it should break loudly with a failed build. At that point,
// someone will need to add another hack into this switch statement, or do the more difficult
// thing and replace this with better scala integration.
//
// If you replace this with better scala integration that follows the fact dictionary,
// this can be wished a nice retirement.
export function unmapFactAlias(path: string): Path {
  let alias: string | undefined;
  // all address alises get undone;
  if (belongsToAlias(path, addressFields)) {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    return addressFields.find((af) => path.startsWith(af))!;
  } else if (belongsToAlias(path, bankAccountFields)) {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    return bankAccountFields.find((f) => path.startsWith(f))!;
  } else if (path.startsWith(`/formW2s/*/filer`)) {
    // we call the function twice, since we may end with an ssn functino
    // or a member of a collection
    return unmapFactAlias(path.replace(`/formW2s/*/filer`, `/filers/*`));
  } else if (path.startsWith(`/hsaDistributions/*/filer`) && path !== `/hsaDistributions/*/filer`) {
    return unmapFactAlias(path.replace(`/hsaDistributions/*/filer`, `/filers/*`));
  } else if ([`/month`, `/year`, `/day`].some((dateSelector) => path.endsWith(dateSelector))) {
    // remove day operations
    return unmapFactAlias(path.split(`/`).slice(0, -1).join(`/`) as Path);
  } else if (path.endsWith(`/*`)) {
    // we're loking at a member of a collection, so we look to the collection definition instead
    return path.split(`/`).slice(0, -1).join(`/`) as Path;
  } else if (tinSubfields.some((subfield) => path.endsWith(subfield))) {
    // remove TIN operations and unalias parent, if relevant
    return unmapFactAlias(path.split(`/`).slice(0, -1).join(`/`) as Path);
  } else if (belongsToAlias(path, individualFilerAliases)) {
    return `/filers/*/${path.split(`/`).slice(-1)}` as Path;
    // deal with filter collections
  } else if (belongsToAlias(path, dependentCollectionAliases)) {
    return `/familyAndHousehold/*/${path.split(`/`).slice(-1)}` as Path;
  } else if ((alias = individualDependentAliases.find((alias) => path.startsWith(alias)))) {
    return path.replace(alias, `/familyAndHousehold/*/`) as Path;
  } else if (belongsToAlias(path, filerCollectionAliases)) {
    return `/filers/*/${path.split(`/`).slice(-1)}` as Path;
  } else if (belongsToAlias(path, w2Aliases)) {
    return `/formW2s/*/${path.split(`/`).slice(-1)}` as Path;
  } else if ((alias = individualW2Aliases.find((alias) => path.startsWith(alias)))) {
    return path.replace(alias, `/formW2s/*/`) as Path;
  } else if (belongsToAlias(path, form1099GAliases)) {
    return `/form1099Gs/*/${path.split(`/*/`).slice(-1)}` as Path;
  } else if (belongsToAlias(path, form1099MiscAliases)) {
    return `/form1099Miscs/*/${path.split(`/*/`).slice(-1)}` as Path;
  } else if (belongsToAlias(path, form1099RAliases)) {
    return `/form1099Rs/*/${path.split(`/*/`).slice(-1)}` as Path;
  } else if (belongsToAlias(path, interestReportsAliases)) {
    return `/interestReports/*/${path.split(`/*/`).slice(-1)}` as Path;
  } else if (belongsToAlias(path, hsaDistributionsAliases)) {
    return `/hsaDistributions/*/${path.split(`/*/`).slice(-1)}` as Path;
  } else {
    return path as Path;
  }
}

const belongsToAlias = (path: string, aliasList: readonly string[] | string[]): boolean =>
  aliasList.some((alias) => path.startsWith(alias));

/**
 * The dependencyMap only returns one level deep -- e.g.
 * that a -> b. But it's possible that b -> c, and c -> d, etc. This function follows
 * the dependency map to get all deep dependencies.
 */
export function getDeepDependencies(path: Path, dependencyMap: ReturnType<typeof generateDependencyGraph>): Set<Path> {
  const allDeps = new Set<Path>();
  const stack: Path[] = [path];
  let head = stack.pop();
  while (head) {
    allDeps.add(head);
    const dependencies = dependencyMap.get(head) || [];
    // We only add/check deps that we haven't already seen
    stack.push(...dependencies.map((dep) => dep.path).filter((dep) => !allDeps.has(dep)));
    head = stack.pop();
  }
  return allDeps;
}
