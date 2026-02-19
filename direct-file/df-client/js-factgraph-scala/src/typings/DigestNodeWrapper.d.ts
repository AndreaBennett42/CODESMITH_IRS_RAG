interface BaseWrappedFact {
  path: any;
}

export interface DerivedWrappedFact extends BaseWrappedFact {
  writable: null;
  derived: CompNodeConfigDigestWrapper;
  placeholder: CompNodeConfigDigestWrapper | null;
}

export interface WritableWrappedFact extends BaseWrappedFact {
  writable: WritableOption;
  derived: null;
  placeholder: CompNodeConfigDigestWrapper | null;
}

export type WrappedFact = DerivedWrappedFact | WritableWrappedFact | never;

export declare class DigestNodeWrapper {
  constructor(
    path: string,
    writableOption: WritableOption | null,
    derivedOption: CompNodeConfigDigestWrapper | null,
    placeholderOption: CompNodeConfigDigestWrapper | null
  );
}
// TODO (Scala -> TS Types)
export const DigestNodeWrapperFactory: any;

/**
   * Example 
   * 
       {
          "path": "/filers/star/middleInitial",
          "writable": {
              "typeName": "String",
              "options": {},
              "collectionItemAlias": null,
              "limits": []
          },
          "derived": null,
          "placeholder": null
      },
   * 
   */
export interface WritableOption {
  typeName: string;
  options: { [key: string]: string }; // TODO: I think this is empty after enum refactor
  collectionItemAlias: string | null;
  limits: LimitConfig[];
}

/** Example
   *             "limits": [
                  {
                      "operation": "MaxLength",
                      "level": "Error",
                      "node": {
                          "typeName": "Int",
                          "options": {
                              "value": "35"
                          },
                          "children": []
                      }
                  }
              ]
   */

export interface LimitConfig {
  operation: string;
  level: 'Error' | 'Warn';
  node: {
    typeName: string; // TODO: explicit set of strings;
    options: { [key: string]: string };
    children: CompNodeConfigDigestWrapper[];
  };
}

/**
   * Example 
   * 
   *     {
          "path": "/filers/star/firstNameAndInitial",
          "writable": null,
          "derived": {
              "typeName": "Paste",
              "options": {},
              "children": [
                  {
                      "typeName": "Dependency",
                      "options": {
                          "path": "../firstName"
                      },
                      "children": []
                  },
                  {
                      "typeName": "Dependency",
                      "options": {
                          "path": "../middleInitial"
                      },
                      "children": []
                  }
              ]
          },
          "placeholder": null
      },
   */
export interface CompNodeConfigDigestWrapper {
  typeName: string;
  children: CompNodeConfigDigestWrapper[];
  options: { [key: string]: string };
}
