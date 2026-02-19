import { generateDependencyGraph, unmapFactAlias } from './generate-src/dependencyGraph.js';
import { Path } from './Path.js';
import { facts } from './generated/facts.js';
import { readMeFFactPaths } from './generate-src/readMeFFactPaths.js';
import { format } from 'prettier';

interface FactInModule {
  exportedForDownstreamFacts: boolean;
  exportedForMef: boolean;
  dependencies: { module?: string; path: Path }[];
}

interface MissingModuleDependencyError {
  type: 'MISSING_MODULE';
  missingModule: string;
  requiringFact: string;
  requiringModule: string;
}

interface FactDoesNotExistInModuleError {
  type: 'FACT_DOES_NOT_EXIST_IN_MODULE';
  requiringFact: string;
  requiringModule: string;
  requiredFact: string;
  requiredModule: string;
}

interface FactNotExportedForDownstreamModuleError {
  type: 'FACT_NOT_EXPORTED_FOR_DOWNSTREAM_MODULE';
  requiringFact: string;
  requiringModule: string;
  requiredFact: string;
  requiredModule: string;
}

interface FactDoesNotExistLocallyError {
  type: 'FACT_DOES_NOT_EXIST_LOCALLY';
  requiringFact: string;
  requiringModule: string;
  requiredFact: string;
}

type ModuleError =
  | MissingModuleDependencyError
  | FactDoesNotExistInModuleError
  | FactNotExportedForDownstreamModuleError
  | FactDoesNotExistLocallyError;

/**
 * This function will throw an error if any fact in the fact dictionary:
 * 1. Depends on a fact that does not exist (either locally or across modules)
 * 2. Depends on a module that does not exist
 * 3. Depends on a fact from a different module that is not exported for use
 *    in downstream facts.
 *
 * We run this function in CI as a validator on our fact dictionary to make sure that modules
 * are exported and line up appropriately.
 *
 * It would be better if we had additional tooling that gave us red squigglies in our IDEs.
 */
function verifyModuleDependencies() {
  const moduleMapping = buildModuleMapping();
  const errors: ModuleError[] = [];
  for (const [module, thisModuleFacts] of Object.entries(moduleMapping)) {
    for (const [factName, factInfo] of Object.entries(thisModuleFacts)) {
      for (const dependency of factInfo.dependencies) {
        if (dependency.module) {
          // this dependency must be from another module, so we look up that module and check
          // that a fact there exists and is exported
          const requiredModule = moduleMapping[dependency.module];
          // eslint-disable-next-line eqeqeq
          if (requiredModule == undefined) {
            errors.push({
              type: `MISSING_MODULE`,
              missingModule: dependency.module,
              requiringFact: factName,
              requiringModule: module,
            });
            break;
          }
          const requiredFact = requiredModule[dependency.path];
          // eslint-disable-next-line eqeqeq
          if (requiredFact == undefined) {
            errors.push({
              type: `FACT_DOES_NOT_EXIST_IN_MODULE`,
              requiringFact: factName,
              requiringModule: module,
              requiredFact: dependency.path,
              requiredModule: dependency.module,
            });
          } else if (!requiredFact.exportedForDownstreamFacts) {
            errors.push({
              type: `FACT_NOT_EXPORTED_FOR_DOWNSTREAM_MODULE`,
              requiringFact: factName,
              requiringModule: module,
              requiredFact: dependency.path,
              requiredModule: dependency.module,
            });
          }
        } else {
          // this dependency is in our current module, so we check that it exists here
          const requiredFact = thisModuleFacts[dependency.path];
          // eslint-disable-next-line eqeqeq
          if (requiredFact == undefined) {
            errors.push({
              type: `FACT_DOES_NOT_EXIST_LOCALLY`,
              requiringFact: factName,
              requiringModule: module,
              requiredFact: dependency.path,
            });
          }
        }
      }
    }
  }
  if (errors.length > 0) {
    // eslint-disable-next-line no-console
    console.error(
      `Found the following namespace errors when evaluating the fact dictionary: ${format(JSON.stringify(errors), {
        parser: `json`,
      })}

      If you've added new collection aliases (either via Filter or other means), you may need to modify 'aliases.ts'`
    );
    // TODO: also note exports that are not used by other modules that should be private.
    throw new Error(`${errors.length} Errors were discovered`);
  }
}

function verifyExportedVariablesAreUsedByOtherModules() {
  // eslint-disable-next-line eqeqeq
  const exportedFactPaths = facts.filter((fact) => fact.Export?.[`@downstreamFacts`] == `true`).map((f) => f[`@path`]);
  const moduleMapping = buildModuleMapping();
  const usedExportedFacts = new Set<string>();
  for (const [_module, moduleFacts] of Object.entries(moduleMapping)) {
    for (const [_moduleFactPath, moduleFact] of Object.entries(moduleFacts)) {
      for (const dependency of moduleFact.dependencies) {
        if (dependency.module) {
          usedExportedFacts.add(dependency.path);
        }
      }
    }
  }
  const unusedExportedFactPaths = exportedFactPaths.filter((p) => !usedExportedFacts.has(p));
  if (unusedExportedFactPaths.length > 0) {
    throw new Error(
      `The following facts were exported but not used by other modules:\n${unusedExportedFactPaths.join(`\n`)}`
    );
  }
}

function verifyMefExportedFacts() {
  // eslint-disable-next-line eqeqeq
  const exportedFactPaths = facts.filter((fact) => fact.Export?.[`@mef`] == `true`).map((f) => f[`@path`]);
  const exportedFactPathsSet = new Set(exportedFactPaths);
  // There is a non-critical bug where we're losing _some_ information on unaliasing -- if you have /primaryFiler/name,
  // we transform that to /filers/*/name... so we lose the dependency on /filers/*/isPrimaryFiler
  const factPathsUsedByMef = readMeFFactPaths().map((f) => unmapFactAlias(f));
  const factsUsedByMefSet = new Set(factPathsUsedByMef);
  const factsMissingExports = factPathsUsedByMef.filter((f) => !exportedFactPathsSet.has(f) && !f.includes(`~`));

  if (factsMissingExports.length > 0) {
    throw new Error(
      `The following facts were required by MeF but not exported in the fact dictionary:\n${factsMissingExports.join(
        `\n`
      )}`
    );
  }
  const unusedExportedFactPaths = exportedFactPaths.filter((p) => !factsUsedByMefSet.has(p as Path));
  if (unusedExportedFactPaths.length > 0) {
    throw new Error(
      `The following facts were exported to MeF but not used by MeF:\n${unusedExportedFactPaths.join(`\n`)}`
    );
  }
}

/**
 *
 * @returns a map of which facts exist in which modules, and for each fact, whether it is exported
 *          and what its dependencies are.
 */
export function buildModuleMapping() {
  const dependencyGraph = generateDependencyGraph();
  const moduleFacts: { [module: string]: { [factName: string]: FactInModule } } = {};
  for (const fact of facts) {
    const moduleName = fact.srcFile.replace(`.xml`, ``);
    if (!moduleFacts[moduleName]) {
      moduleFacts[moduleName] = {};
    }
    const factsInModule = moduleFacts[moduleName];
    const dependencies = dependencyGraph.get(fact[`@path`] as Path);
    if (dependencies === undefined) {
      throw new Error(`Fact ${fact[`@path`]} did not exist in the dependency graph`);
    }
    factsInModule[fact[`@path`]] = {
      // eslint-disable-next-line eqeqeq
      exportedForDownstreamFacts: fact.Export?.[`@downstreamFacts`] == `true`,
      // eslint-disable-next-line eqeqeq
      exportedForMef: fact.Export?.[`@mef`] == `true`,
      dependencies,
    };
  }
  return moduleFacts;
}

verifyModuleDependencies();
verifyExportedVariablesAreUsedByOtherModules();
verifyMefExportedFacts();
