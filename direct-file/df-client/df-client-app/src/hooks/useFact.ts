import { ConcretePath, FactGraph, FactGraphResult } from '@irs/js-factgraph-scala';
import { useFactGraph } from '../factgraph/FactGraphContext.js';
import { useCallback, useEffect, useRef, useState } from 'react';
import { FACT_PATH_TYPES } from '../factgraph/factGraphTypeHelpers.js';
import { Path } from '../flow/Path.js';
import { InterceptingFactGraph } from '../factgraph/InterceptingFactGraph.js';
import { TaxReturn } from '../types/core.js';

export const getFactFromTaxReturn = <T>(path: ConcretePath, taxReturn: TaxReturn) => {
  const factGraph = new InterceptingFactGraph(taxReturn.facts);
  return getFactFromGraph<T>(path, factGraph);
};

const getFactFromGraph = <T>(path: ConcretePath, factGraph: FactGraph) => {
  const fact: FactGraphResult<T> = factGraph.get(path);
  const currentValue = fact.hasValue ? fact.get : undefined;
  const isComplete = fact.complete;

  return [currentValue, isComplete] as const;
};

type FactState<T> = readonly [T | undefined, boolean];
export type UseFactResult<T> = readonly [T | undefined, (value: T) => void, () => void, boolean];
export function useSingleFact<T>(path: ConcretePath) {
  const { factGraph } = useFactGraph();

  const getFact = useCallback((): FactState<T> => {
    return getFactFromGraph(path, factGraph);
  }, [factGraph, path]);

  const setFact = useCallback(
    (value: T) => {
      factGraph.set(path, value);
    },
    [factGraph, path]
  );

  const clearFact = useCallback(() => {
    factGraph.delete(path);
  }, [path, factGraph]);

  return [getFact, setFact, clearFact] as const;
}

function useFact<T>(path: ConcretePath): UseFactResult<T> {
  const [getFact, setFact, clearFact] = useSingleFact<T>(path);

  const [currentValue, isComplete] = getFact();

  return [currentValue, setFact, clearFact, isComplete];
}

export function useDynamicFact<T>(path: ConcretePath | undefined): UseFactResult<T> {
  const { factGraph } = useFactGraph();
  const [getFact, setFact, clearFact] = useSingleFact<T>(path as ConcretePath);

  // We persistently track the current value so we can move it when the path changes
  const [currentState, setCurrentState] = useState<FactState<T>>(
    path === undefined ? ([undefined, false] as FactState<T>) : getFact()
  );

  const [currentFactType, setFactType] = useState(path ? FACT_PATH_TYPES.get(Path.fromConcretePath(path)) : undefined);

  const [currentValue, isComplete] = currentState;

  const setValue = useCallback(
    (value: T) => {
      // Note: Our typings should prevent use from ever getting an undefined value, but we still check as a precaution
      setCurrentState([value, value !== undefined]);
      if (path !== undefined) setFact(value);
    },
    [path, setFact]
  );

  const clearValue = useCallback(() => {
    setCurrentState([undefined, false]);
    if (path !== undefined) clearFact();
  }, [clearFact, path]);

  const previousPathRef = useRef(path);

  useEffect(() => {
    if (path === previousPathRef.current) return;

    const newFactType = path ? FACT_PATH_TYPES.get(Path.fromConcretePath(path)) : undefined;

    // Clear the value at the previous path
    if (previousPathRef.current !== undefined) {
      factGraph.delete(previousPathRef.current);
    } else clearValue();

    // Set the current value to the updated path
    if (path !== undefined && currentValue !== undefined) {
      if (newFactType === currentFactType) {
        // The new fact's type matches the previous
        setFact(currentValue);
      } else {
        // The new fact's type is different.  Discard the previous value and update the fact type
        setFactType(newFactType);
      }
    }

    // else, there is no value to carry over
    previousPathRef.current = path;
  }, [clearValue, currentFactType, currentValue, factGraph, isComplete, path, setFact]);

  return [currentValue, setValue, clearValue, isComplete];
}

export default useFact;
