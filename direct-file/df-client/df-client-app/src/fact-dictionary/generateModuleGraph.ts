import { fileURLToPath } from 'url';
import { buildModuleMapping } from './verifyModuleDependencies.js';
import fs from 'fs';

interface Edge {
  sourceModule: string;
  destinationModule: string;
  factPath: string;
}

function stringifyEdge(e: Edge) {
  return `${e.destinationModule}${e.sourceModule}${e.factPath}`;
}

function generateModuleGraph() {
  const edges: Edge[] = [];
  const seenEdges = new Set<string>();
  const moduleMapping = buildModuleMapping();
  for (const [module, moduleFacts] of Object.entries(moduleMapping)) {
    for (const [_requiringFact, moduleFact] of Object.entries(moduleFacts)) {
      for (const dependency of moduleFact.dependencies) {
        if (dependency.module) {
          const edge = {
            sourceModule: dependency.module,
            destinationModule: module,
            factPath: dependency.path,
          };
          const edgeString = stringifyEdge(edge);
          if (!seenEdges.has(edgeString)) {
            edges.push(edge);
            // Seen Edges acts to deduplicate our edges. It's like a java HashSet, but jankier because it's javascript.
            seenEdges.add(edgeString);
          }
        }
      }
    }
  }

  const exportedFacts = new Set(edges.map((e) => e.factPath));

  const digraphEdgesString = edges.map((e) => `${e.sourceModule} -> ${e.destinationModule} [ label="${e.factPath}" ]`);
  const dotString = `
digraph {
label="${Object.keys(moduleMapping).length} modules connected by ${exportedFacts.size} exported facts"
labelloc=top;
labeljust=left;
  ${digraphEdgesString.join(`\n`)}
}
  `;
  // Writes a dotfile. Transform into an svg using graphviz.
  fs.writeFileSync(`moduleGraph.dot`, dotString);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  generateModuleGraph();
}
