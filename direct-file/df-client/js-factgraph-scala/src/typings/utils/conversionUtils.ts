export interface ScalaOptional<T> {}
export interface ScalaList<T> {}
export interface ScalaSet<T> {}
export interface ScalaMap<K, V> {}

export declare function unwrapScalaOptional<T>(opt: ScalaOptional<T>): T | null;
export declare function scalaListTArray<T>(list: ScalaList<T>): T[];
export declare function scalaMapTMap<K, V>(map: ScalaMap<K, V>): Map<K, V>;
export declare function ArrayToScalaList<T>(list: T[]): ScalaList<T>;
export declare function jsSetToScalaSet<T>(set: Set<T>): ScalaSet<T>;
export declare function scalaSetToJsSet<T>(set: ScalaSet<T>): Set<T>;
