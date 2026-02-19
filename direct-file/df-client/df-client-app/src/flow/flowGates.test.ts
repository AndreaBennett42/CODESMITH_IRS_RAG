import { describe, it, expect } from 'vitest';
import { createFlowConfig } from './flowConfig.js';
import flowNodes from './flow.js';
import { Condition, RawCondition } from './Condition.js';
import { contentConfigIsFactConfig } from './ContentDeclarations.js';
import { generateDependencyGraph, getDeepDependencies } from '../fact-dictionary/generate-src/dependencyGraph.js';
import { Path } from '../fact-dictionary/Path.js';

// right now, this is a condition on every screen, and is almost always irrecoverable.
const EXCLUDED_GATE_CONDITIONS: RawCondition[] = [{ operator: `isFalseOrIncomplete`, condition: `/flowIsKnockedOut` }];

// These screens are excluded because they may intentionally override their gate status.
// They are currently all specific to filing status -- since we show different filing status screens
// based on whether you are there for the first time or not, and whether you have an invalid filing status.
const EXCLUDED_SCREENS = [`filing-status-override`, `filing-status-error-manual-fix`, `hoh-qp-selection`];

describe(`Writable facts in the flow are not gated by their own value`, () => {
  const flow = createFlowConfig(flowNodes);
  const screensWithWritableFacts = flow.screens.filter((s) => {
    const factContent = s.content
      .filter(contentConfigIsFactConfig)
      // eslint-disable-next-line eqeqeq
      .filter((c) => !c.props.readOnly && c.props.displayOnlyOn != `data-view`);
    return factContent.length > 0;
  });
  const dependencyMap = generateDependencyGraph();
  for (const screen of screensWithWritableFacts) {
    if (EXCLUDED_SCREENS.includes(screen.route)) {
      continue;
    }
    const { route, conditions } = screen;
    const factPaths = screen.content
      .filter(contentConfigIsFactConfig)
      // eslint-disable-next-line eqeqeq
      .filter((c) => !c.props.readOnly && c.props.displayOnlyOn != `data-view`)
      .map((c) => c.props.path);
    it(`Screen ${route} with facts ${factPaths.join(`,`)} is not gated by any of its values`, () => {
      const allDependencies = new Set<Path>();
      conditions.forEach((c) => {
        // JSON.stringify gives us deep equality in case the condition is an object
        // eslint-disable-next-line eqeqeq
        if (EXCLUDED_GATE_CONDITIONS.find((ex) => JSON.stringify(ex) == JSON.stringify(c))) {
          return;
        }
        const factPath = new Condition(c).innerCondition.factPath;
        if (factPath) {
          const factDeps = getDeepDependencies(factPath, dependencyMap);
          factDeps.forEach((dep) => allDependencies.add(dep));
        }
      });
      for (const factPath of factPaths) {
        expect(allDependencies, `${route} contained ${factPath} in a gate condition`).not.toContain(factPath);
      }
    });
  }
});
