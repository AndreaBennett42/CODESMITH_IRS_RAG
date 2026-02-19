import { DependencyList, EffectCallback, useEffect, useRef } from 'react';

type DependencyComparisons = Record<string, DependencyComparison>;

type DependencyComparison = {
  before: unknown;
  after: unknown;
};

const usePrevious = <T>(value: T, initialValue: T) => {
  const ref = useRef(initialValue);
  useEffect(() => {
    ref.current = value;
  });
  return ref.current;
};

/**
 A debugging utility that helps identify which dependencies have changed.
 Useful for triaging unexpected renders or useEffect hook runs

 Simply take any existing useEffect and replace it with useEffectDebugger to use.
 Optionally, add a third argument for the dependency names for a better debugging experience

 e.g.

 from

 ```
 useEffect(() => {
   // do stuff...
 }, [dep1, dep2])
 ```

 to

 ```
 useEffectDebugger(() => {
 // do stuff...
 }, [dep1, dep2])
 ```

 or

 ```
 useEffectDebugger(() => {
 // do stuff...
 }, [dep1, dep2], ["dep1", "dep2"])
 ```
 */
export const useEffectDebugger = (effectHook: EffectCallback, dependencies: DependencyList, dependencyNames = []) => {
  const previousDeps = usePrevious(dependencies, []);

  const changedDeps = dependencies.reduce<DependencyComparisons>((accum, dependency, index) => {
    if (dependency !== previousDeps[index]) {
      const keyName = dependencyNames[index] || index;
      return {
        ...accum,
        [keyName]: {
          before: previousDeps[index],
          after: dependency,
        },
      };
    }

    return accum;
  }, {});

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(effectHook, dependencies);
};
