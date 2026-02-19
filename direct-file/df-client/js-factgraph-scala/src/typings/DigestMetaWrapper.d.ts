export interface EnumDeclaration {
  id: string;
  options: string[];
}

export declare class DigestMetaWrapper {
  constructor(year: string);
  toNative: () => DigestMetaNative;
}

export interface DigestMetaNative {}
