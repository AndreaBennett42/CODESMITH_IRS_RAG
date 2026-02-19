import { fileURLToPath } from 'url';
import { Dependency, generateDependencyGraph } from './generate-src/dependencyGraph.js';
import { wrappedFacts } from './generated/wrappedFacts.js';
import fs from 'fs';

function main() {
  const dependencyGraph = generateDependencyGraph();
  const digraphEdgesString = Object.entries(Object.fromEntries(dependencyGraph)).flatMap(
    ([key, deps]: [string, Dependency[]]) => {
      return deps.map((dep) => `"${key}" -> "${dep.path}"`);
    }
  );
  const dotString = `
strict digraph {
  label="${wrappedFacts.length} facts connected by ${digraphEdgesString.length} dependencies"
  labelloc=top;
  labeljust=left;
  ${digraphEdgesString.join(`\n`)}
}
  `;
  // Writes a dotfile. Transform into an svg using graphviz.
  fs.writeFileSync(`dependencyGraph.dot`, dotString);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main();
}
