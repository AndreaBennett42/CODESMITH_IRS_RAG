/* eslint-disable no-console */
import { File, Task, Test } from 'vitest';
import { createFlowConfig } from '../flow/flowConfig.js';
import fs from 'fs';
import flowNodes from '../flow/flow.js';
import * as child from 'child_process';
import { Path } from '../fact-dictionary/Path.js';
import { facts } from '../fact-dictionary/generated/facts.js';

/**
 * Welcome to our custom tax logic test coverage reporter!
 *
 * This reporter exists to report the coverage metrics described by the tax logic testing strategy
 * described in /docs/adr/tax-logic-testing-strategy.md. Namely it reports on which screens have
 * functional flow tests, and which derived facts in the fact dictionary have been tested, with an
 * ability to focus on culminating facts and mef exported facts.
 *
 * It's important that these metrics remain metrics, and not targets. These metrics could be very easily
 * manipulated to increase their number without accomplishing their goal. E.g. you could write one test
 * for "dependents/ASTERISK/qualifyingChild" and it would mark that as a tested fact -- but the developer
 * writing the test for qualifyingChild should realize they have to write at least five tests for the five
 * criteria it takes to be a qualifying child. Similarly, a functional flow test could only follow one path
 * from a screen, when there may be multiple paths to test.
 *
 * These tests are certainly useful in January 2024 as we try to measure + assure our quality prior to launch.
 * In the future, it may make sense to replace or extend these with other metrics that can prove not just that
 * tests for a fact or screen exist, but that the tests are comprehensive.
 */

const COVERAGE_DIR = `./coverage`;
const KNOCKOUT_CATEGORY_ROUTE = `/flow/knockout`;

export default class TaxLogicTestCoverageReporter {
  onFinished(files?: File[]) {
    const visitedFactPaths = new Set<Path>();
    const startingScreenRoutes = new Set<string>();
    if (files) {
      files.forEach((file) => {
        collectTests(file).forEach((task) => {
          const meta = task.meta;
          if (meta.testedFactPaths) {
            meta.testedFactPaths.forEach((p) => visitedFactPaths.add(p));
          }
          if (meta.functionalFlowTestStartingScreenRoutes) {
            meta.functionalFlowTestStartingScreenRoutes.forEach((r) => startingScreenRoutes.add(r));
          }
        });
      });
    }
    reportTestedFactPaths(visitedFactPaths);
    reportFunctionalFlowRoutes(startingScreenRoutes);
    return Promise.resolve();
  }
}

function reportFunctionalFlowRoutes(testedRoutes: Set<string>) {
  const flow = createFlowConfig(flowNodes);
  const coverage = flow.categories
    // We never test coverage for knockouts because you can never navigate away from a knockout
    // eslint-disable-next-line eqeqeq
    .filter((c) => c.route != KNOCKOUT_CATEGORY_ROUTE)
    .flatMap((c) =>
      c.subcategories.map((sc) => {
        const subcategoryRoute = sc.route;
        const testedScreens = sc.screens.filter((sc) => testedRoutes.has(sc.screenRoute));
        const untestedScreens = sc.screens.filter((sc) => !testedRoutes.has(sc.screenRoute));
        return {
          subcategoryRoute,
          testedScreens,
          untestedScreens,
          totalRoutes: testedScreens.length + untestedScreens.length,
        };
      })
    );

  const totalCounts = coverage.reduce(
    (prevValue, subcategory) => {
      prevValue.testedScreens += subcategory.testedScreens.length;
      prevValue.totalScreens += subcategory.totalRoutes;
      return prevValue;
    },
    { totalScreens: 0, testedScreens: 0 }
  );
  console.log(
    `${totalCounts.testedScreens}/${totalCounts.totalScreens} screens have been tested for functional flow coverage`
  );

  const COVERAGE_FILE = `${COVERAGE_DIR}/functionalFlowCoverage.html`;
  if (!fs.existsSync(COVERAGE_DIR)) {
    fs.mkdirSync(COVERAGE_DIR);
  }
  if (fs.existsSync(COVERAGE_FILE)) {
    fs.rmSync(COVERAGE_FILE);
  }
  const revision = child.execSync(`git rev-parse HEAD`).toString().trim();
  // for easy consumption from circleci, we write this coverage report as HTML without css
  const htmlContents = `
  <html>
  <body>
  <h1>Functional Flow Test Coverage</h1>
  <p>Revision ${revision}</p>
  <p>Date ${new Date().toLocaleString()}</p>
  ${coverage
    .map((subcat) => {
      return `
      <h2>${subcat.subcategoryRoute}</h2>
      <h3>Tested Screens ${subcat.testedScreens.length}/${subcat.totalRoutes}</h3>
      <ul>
      ${subcat.testedScreens.map((sc) => `<li>${sc.screenRoute}</li>`).join(`\n`)}
      </ul>
      <h3>Untested Screens ${subcat.untestedScreens.length}/${subcat.totalRoutes}</h3>
      <ul>
      ${subcat.untestedScreens.map((sc) => `<li>${sc.screenRoute}</li>`).join(`\n`)}
      </ul>
    `;
    })
    .join(`\n`)}
  </body>
  </html>
  `;
  fs.writeFileSync(COVERAGE_FILE, htmlContents);
}

function reportTestedFactPaths(visitedFactPaths: Set<string>) {
  const derivedFactsByModule = facts.reduce(
    (ret: { [srcFile: string]: { path: Path; usedByMef: boolean; culminating: boolean }[] }, fact) => {
      if (!ret[fact.srcFile]) {
        ret[fact.srcFile] = [];
      }
      if (fact.Derived) {
        ret[fact.srcFile].push({
          path: fact[`@path`] as Path,
          usedByMef: Boolean(fact.Export?.[`@mef`]),
          culminating: Boolean(fact.Export?.[`@downstreamFacts`]),
        });
      }
      return ret;
    },
    {}
  );
  const coverage = Object.entries(derivedFactsByModule).map(([srcFile, facts]) => {
    const mefFacts = facts.filter((f) => f.usedByMef);
    const culminatingFacts = facts.filter((f) => f.culminating);
    const testedFacts = facts.filter((f) => visitedFactPaths.has(f.path));
    const untestedFacts = facts.filter((f) => !visitedFactPaths.has(f.path));
    const testedMefFacts = mefFacts.filter((f) => visitedFactPaths.has(f.path));
    const untestedMefFacts = mefFacts.filter((f) => !visitedFactPaths.has(f.path));
    const testedCulminatingFacts = culminatingFacts.filter((f) => visitedFactPaths.has(f.path));
    const untestedCulminatingFacts = culminatingFacts.filter((f) => !visitedFactPaths.has(f.path));
    return {
      srcFile,
      testedFacts,
      testedMefFacts,
      testedCulminatingFacts,
      untestedCulminatingFacts,
      mefFacts,
      untestedMefFacts,
      culminatingFacts,
      untestedFacts,
      allFacts: facts,
    };
  });

  const totalCounts = coverage.reduce(
    (prevValue, module) => {
      prevValue.testedCulminatingFacts += module.testedCulminatingFacts.length;
      prevValue.culminatingFacts += module.culminatingFacts.length;
      return prevValue;
    },
    { testedCulminatingFacts: 0, culminatingFacts: 0 }
  );

  console.log(`${totalCounts.testedCulminatingFacts}/${totalCounts.culminatingFacts} culminating facts are tested`);

  const COVERAGE_FILE = `${COVERAGE_DIR}/factDictionaryCoverage.html`;
  if (!fs.existsSync(COVERAGE_DIR)) {
    fs.mkdirSync(COVERAGE_DIR);
  }
  if (fs.existsSync(COVERAGE_FILE)) {
    fs.rmSync(COVERAGE_FILE);
  }
  const revision = child.execSync(`git rev-parse HEAD`).toString().trim();
  // for easy consumption from circleci, we write this coverage report as HTML without css
  const htmlContents = `
  <html>
  <body>
  <h1>Fact Dictionary Test Coverage</h1>
  <p>Revision ${revision}</p>
  <p>Date ${new Date().toLocaleString()}</p>
  ${coverage
    .map((module) => {
      return `
      <h2>${module.srcFile}</h2>
      <h3>Culminating Facts</h3>
      <h4>Tested Facts ${module.testedCulminatingFacts.length}/${module.culminatingFacts.length}</h4>
      <ul>
      ${module.testedCulminatingFacts.map((f) => `<li>${f.path}</li>`).join(`\n`)}
      </ul>
      <h4>Untested Facts ${module.untestedCulminatingFacts.length}/${module.culminatingFacts.length}</h4>
      <ul>
      ${module.untestedCulminatingFacts.map((f) => `<li>${f.path}</li>`).join(`\n`)}
      </ul>

      <h3>MeF Facts</h3>
      <h4>Tested Facts ${module.testedMefFacts.length}/${module.mefFacts.length}</h4>
      <ul>
      ${module.testedMefFacts.map((f) => `<li>${f.path}</li>`).join(`\n`)}
      </ul>
      <h4>Untested Facts ${module.untestedMefFacts.length}/${module.mefFacts.length}</h4>
      <ul>
      ${module.untestedMefFacts.map((f) => `<li>${f.path}</li>`).join(`\n`)}
      </ul>

      <h3>All Facts</h3>
      <h4>Tested Facts ${module.testedFacts.length}/${module.allFacts.length}</h4>
      <ul>
      ${module.testedFacts.map((f) => `<li>${f.path}</li>`).join(`\n`)}
      </ul>
      <h4>Untested Facts ${module.untestedFacts.length}/${module.allFacts.length}</h4>
      <ul>
      ${module.untestedFacts.map((f) => `<li>${f.path}</li>`).join(`\n`)}
      </ul>
    `;
    })
    .join(`\n`)}
  </body>
  </html>
  `;
  fs.writeFileSync(COVERAGE_FILE, htmlContents);
}

function collectTests(task: Task): Test[] {
  if (task.type === `suite`) {
    return task.tasks.flatMap((t) => collectTests(t));
  } else if (task.type === `test`) {
    return [task];
  } else {
    // We're in a custom runner of some sort
    // eslint-disable-next-line no-console
    console.warn(`Custom runner, cannot collect tests for Tax LogicTestCoverageReporter`);
    return [];
  }
}
