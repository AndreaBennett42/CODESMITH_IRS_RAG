import { FactDictionary } from './FactDictionary';
import { Persister } from './Persister';
import { ConcretePath } from './ConcretePath';

// I think this is actually an instance of FactGraph, but I don't want to rename all FactGraph to graph
// TODO (Scala -> TS Types)
export const Graph: any;

export declare const GraphFactory: {
  apply: (factDictionary: FactDictionary, persister: Persister) => FactGraph;
};

export interface LimitViolation {
  limitName: string;
  factPath: ConcretePath;
  level: 'Warn' | 'Error';
  limit: string;
  actual: string;
}

export interface SaveReturnValue {
  valid: boolean;
  limitViolations: LimitViolation[];
}

export interface PersisterSyncIssue {
  path: string;
  message: string;
}

export interface Fact {
  path: string;
  value: any;
  limits: any;
  graph: FactGraph;
  path: any;
  meta: any;
}

export interface FactGraph {
  delete: (path: ConcretePath) => void;
  set: (path: ConcretePath, value: any) => void;
  get: (path: ConcretePath) => FactGraphResult<any>;
  getFact: (path: ConcretePath) => Fact;
  getDictionary: () => FactDictionary;
  save: () => SaveReturnValue;
  toJSON: () => string;
  toJson: () => string;
  toStringDictionary: () => any;
  // For a path that is currently incomplete, get the list of paths that could complete the
  // fact.
  explainAndSolve: (path: ConcretePath) => string[][];
  checkPersister: () => PersisterSyncIssue[];
}

interface BaseFactGraphResult {
  toString: string;
  typeName: string;
  hasValue: boolean;
}

export interface CompleteFactGraphResult<T> extends BaseFactGraphResult {
  complete: boolean;
  get: T;
}

export interface IncompleteFactGraphResult extends BaseFactGraphResult {
  complete: false;
  get: never;
}

export type FactGraphResult<T> = IncompleteFactGraphResult | CompleteFactGraphResult<T>;
