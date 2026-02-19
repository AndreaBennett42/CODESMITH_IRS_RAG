import { Meta } from './Meta';
import { DigestMetaNative } from './DigestMetaWrapper';
import { ScalaOptional } from './utils/conversionUtils';
// TODO (Scala -> TS Types)
export const FactDictionaryFactory: any;

export declare const FactDictionaryConfig: {
  create: (meta: DigestMetaNative, facts: FactsNative) => FactConfig;
  fromConfig: (config: any) => FactDictionary;
  getPaths: () => any[];
  getDefinition: (path: any) => FactDefinition;
};

export interface FactDictionary {
  getMeta: () => Meta;
  getOptionsPathForEnum: (enumPath: any) => ScalaOptional<string>;
  getDefinition: (path: any) => FactDefinition;
}

export interface FactConfig {}
export interface FactsNative {}
