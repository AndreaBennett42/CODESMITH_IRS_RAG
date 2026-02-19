import { generateDependencyGraph } from './generate-src/dependencyGraph.js';
import { Path } from './Path.js';
import { facts } from './generated/facts.js';
import { describe, it } from 'vitest';

// This dependency alway comes up circular due to how we de-alias. It would be better if we were better.
const EXCEPTIONS = [`/formW2s/*/address`, `/form1099Rs/*/address`];

describe(`Circular facts`, () => {
  const dependencyMap = generateDependencyGraph();
  const derivedFacts = facts.filter((f) => f.Derived);
  for (const fact of derivedFacts) {
    it(`${fact[`@path`]} has no circular dependencies`, () => {
      const path = fact[`@path`] as Path;
      const visited = new Set<Path>();
      const stack: Path[][] = [[path]];
      let currentPath = stack.pop();
      while (currentPath) {
        const head = currentPath[0];
        visited.add(head);
        const dependencies = dependencyMap.get(head)?.map((dep) => dep.path) || [];
        for (const dep of dependencies) {
          if (EXCEPTIONS.includes(dep)) {
            continue;
          }
          if (currentPath.includes(dep)) {
            throw new Error(`Circular fact for fact ${path} on path ${[dep, ...currentPath].reverse().join(` -> `)}`);
          }
          if (!visited.has(dep)) {
            stack.push([dep, ...currentPath]);
          }
        }
        currentPath = stack.pop();
      }
    });
  }
});
