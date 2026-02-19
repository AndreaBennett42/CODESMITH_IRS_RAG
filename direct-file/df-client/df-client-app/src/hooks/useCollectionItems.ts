import { unwrapScalaOptional, scalaListToJsArray, FactGraph } from '@irs/js-factgraph-scala';
import { useFactGraph } from '../factgraph/FactGraphContext.js';
import { Path } from '../flow/Path.js';
import { Path as FGPath } from '../fact-dictionary/Path.js';

// XXX: It would be nice if we could type the path to just collection items
export function useCollectionItems(path: FGPath, collectionId: string | null) {
  const { factGraph } = useFactGraph();
  return getCollectionItems(path, collectionId, factGraph);
}

export function getCollectionItems(path: FGPath, collectionId: string | null, factGraph: FactGraph) {
  // looks like we have to use any here, so it's a farrago of ignores to silence all the warnings
  // eslint-disable-next-line
  const collectionPath = unwrapScalaOptional(factGraph.getDictionary().getDefinition(path).value.getAlias()) as  // eslint-disable-next-line @typescript-eslint/no-explicit-any
    | any
    | null;
  if (!collectionPath) {
    throw new Error(`No collection set for CollectionItem ${path}`);
  }
  const concreteCollectionPath = Path.concretePath(collectionPath.toString(), collectionId);
  const result = factGraph.get(concreteCollectionPath);
  const collectionItems: string[] = result.complete ? scalaListToJsArray(result.get.getItemsAsStrings()) : [];
  return { collectionPath: collectionPath.toString() as FGPath, collectionItems };
}
