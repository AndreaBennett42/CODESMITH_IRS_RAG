import { AbstractPath, Path as FDPath } from '../fact-dictionary/Path.js';
import { ConcretePath } from '@irs/js-factgraph-scala';

export class Path {
  static isAbstract(path: FDPath) {
    return path.includes(`*`);
  }

  static isAbstractPathType(path: FDPath): path is AbstractPath {
    return this.isAbstract(path);
  }

  static concretePath(path: FDPath, collectionId: string | null) {
    return path.replace(`*`, `#${collectionId}`) as ConcretePath;
  }

  static fromConcretePath(concretePath: ConcretePath) {
    const regexPattern = /\/#([a-fA-F0-9-]+)\//g;
    return concretePath.replace(regexPattern, `/*/`) as FDPath;
  }
}
