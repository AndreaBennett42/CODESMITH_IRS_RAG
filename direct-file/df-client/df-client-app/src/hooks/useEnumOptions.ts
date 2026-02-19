import { unwrapScalaOptional, scalaListToJsArray, ScalaList, FactGraph } from '@irs/js-factgraph-scala';
import { useFactGraph } from '../factgraph/FactGraphContext.js';
import { Path } from '../flow/Path.js';
import { AbsolutePath, Path as FGPath } from '../fact-dictionary/Path.js';

export function useEnumOptions(path: FGPath, collectionId: string | null) {
  const { factGraph } = useFactGraph();
  return getEnumOptions(factGraph, path, collectionId);
}

export function getEnumOptions(factGraph: FactGraph, path: FGPath, collectionId: string | null) {
  const optionsPath = unwrapScalaOptional(factGraph.getDictionary().getOptionsPathForEnum(path)) as AbsolutePath;
  if (!optionsPath) {
    throw new Error(`No enum options path for ${path}`);
  }
  const concreteOptionsPath = Path.concretePath(optionsPath, collectionId);
  const result = factGraph.get(concreteOptionsPath);
  const values: string[] | undefined = result.complete
    ? scalaListToJsArray(result.get as ScalaList<string>)
    : undefined;

  return { optionsPath, values };
}
