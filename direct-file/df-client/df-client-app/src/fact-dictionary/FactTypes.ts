export interface RawFact {
  [`@path`]: string;
  [`@testOnly`]?: string;
  Name?: string;
  Description?: string;
  ExportZero?: unknown;
  TaxYear?: 2023;
  BlockSubmissionOnTrue?: '';
  Derived?: unknown;
  Writable?: unknown;
  Placeholder?: unknown;
  srcFile: string;
  Export?: {
    '@downstreamFacts'?: 'true';
    '@mef'?: 'true';
    '@stateSystems'?: 'true';
  };
}
