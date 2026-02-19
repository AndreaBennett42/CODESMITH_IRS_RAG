import { paths, writablePaths } from './generated/paths.js';
export type Path = (typeof paths)[number];
export type WritablePath = (typeof writablePaths)[number];
export type AbsolutePath = Path & `/${string}`;
export type WritableAbsolutePath = WritablePath & `/${string}`;
export type RelativePath = (Path & `../${string}`) | `..`;
export type AbstractPath = Path & `${string}*${string}`;
