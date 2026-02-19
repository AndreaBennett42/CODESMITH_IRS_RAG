import fs from 'fs';
import * as prettier from 'prettier';
import stringifyObject from 'stringify-object';
import processFactsToDigestWrapper from './processFactsToDigestWrapper.js';
import readRawFacts from './readRawFacts.js';
import {
  individualFilerAliases,
  w2Aliases,
  w2Slices,
  form1099MiscAliases,
  form1099RAliases,
  dependentCollectionAliases,
  individualDependentAliases,
  bankAccountFields,
  addressFields,
  filerFields,
  aliasPaths,
  filerCollectionAliases,
} from '../aliases.js';
import { addressSubfields, filerSubfields, tinSubfields } from './subfields.js';
const generatedWarning = `
/**
 * DO NOT MODIFY DIRECTLY. GENERATED FILE.
 * MODIFY \`generate-src/generate.ts\` INSTEAD
 */
`;

const rawFacts = readRawFacts();

const rawFactInterfaces = `
import { RawFact } from "../FactTypes.js";
`;
const factsFileName = `./src/fact-dictionary/generated/facts.ts`;
const factsString = `${generatedWarning} ${rawFactInterfaces} 
// ts-ignore as the typescript compiler cannot check a literal type with over 1000 facts
// because we cast the export again, the export is type safe.
// the only unsafe thing would be if the generator built something that did not match the WrappedFact interface
// @ts-ignore
const tsIgnoredFacts: RawFact[] = ${stringifyObject(rawFacts)} 
export const facts = tsIgnoredFacts as RawFact[] 
`;
const prettyFactsString = prettier.format(factsString, {
  parser: `typescript`,
});

fs.writeFileSync(factsFileName, prettyFactsString);

const paths = rawFacts.map((f) => f[`@path`]);
const writablePaths = rawFacts.filter((f) => f.Writable !== undefined).map((f) => f[`@path`]);

const getExtendedPaths = (paths: string[]) => {
  return Array.from(
    new Set([
      ...paths,
      ...aliasPaths,

      // Hack until we have the types broken out a little better
      // This got out of hand faster than I expected it to
      // Aliases
      ...paths
        .filter((p: string) => p.includes(`filers`))
        .flatMap((p: string) => individualFilerAliases.map((filerAlias) => `${filerAlias}${p.split(`/`).slice(-1)}`)),

      ...paths
        .filter((p: string) => p.includes(`filers`))
        .flatMap((p: string) => filerCollectionAliases.map((filerAlias) => `${p.replace(`/filers/`, filerAlias)}`)),

      ...paths
        .filter((p: string) => p.includes(`familyAndHousehold`))
        .flatMap((p: string) => individualDependentAliases.map((alias) => `${alias}${p.split(`/`).slice(-1)}`)),
      ...paths
        .filter((p: string) => p.includes(`familyAndHousehold`))
        .flatMap((p: string) => dependentCollectionAliases.map((alias) => `${alias}*/${p.split(`/`).slice(-1)}`)),

      ...paths
        .filter((p: string) => p.includes(`form1099Miscs`))
        .flatMap((p: string) =>
          form1099MiscAliases.map((form1099MiscAlias) => `${p.replace(`/form1099Miscs/`, form1099MiscAlias)}`)
        ),

      ...paths
        .filter((p: string) => p.includes(`form1099Rs`))
        .flatMap((p: string) =>
          form1099RAliases.map((form1099RAlias) => `${p.replace(`/form1099Rs/`, form1099RAlias)}`)
        ),

      ...paths
        .filter((p: string) => p.includes(`formW2s`))
        .flatMap((p: string) => w2Aliases.map((w2Alias) => `${w2Alias}*/${p.split(`/`).slice(-1)}`)),

      ...paths
        .filter((p: string) => p.includes(`formW2s`))
        .flatMap((p: string) => w2Slices.map((w2Slice) => `${w2Slice}${p.split(`/`).slice(-1)}`)),

      // Date subpaths
      ...paths.filter((p: string) => p.includes(`dateOfBirth`)).flatMap((p: string) => [`${p}/year`]),
      ...paths.filter((p: string) => p.includes(`dateOfBirth`)).flatMap((p: string) => [`${p}/month`]),
      ...paths.filter((p: string) => p.includes(`dateOfBirth`)).flatMap((p: string) => [`${p}/day`]),

      // TIN subpaths
      ...paths
        .filter((p: string) => p.split(`/`).slice(-1)[0] === `tin`)
        .flatMap((p: string) => {
          return tinSubfields.map((subfield) => `${p}/${subfield}`);
        }),

      // Filer Tin Subpaths
      ...paths
        .filter((p: string) => individualFilerAliases.includes(p.split(`/`).slice(-1)[0]))
        .flatMap((p: string) => tinSubfields.map((subfield) => `${p}/tin/${subfield}`)),

      // Address subpaths
      ...paths
        .filter((p: string) => p.split(`/`).slice(-1)[0] === `address`)
        .flatMap(() => addressFields.flatMap((field) => addressSubfields.map((subfield) => `${field}/${subfield}`))),

      // Bank account subpaths
      ...bankAccountFields.flatMap((p) => [`${p}/accountType`, `${p}/accountNumber`, `${p}/routingNumber`]),

      // Filer subpaths
      ...paths
        .filter((p: string) => p.split(`/`).slice(-1)[0] === `filer`)
        .flatMap(() => filerFields.flatMap((field) => filerSubfields.map((subfield) => `${field}/${subfield}`))),
    ])
  );
};

const extendedPaths = getExtendedPaths(paths);
const extendedWritablePaths = getExtendedPaths(writablePaths);
const pathsFileName = `./src/fact-dictionary/generated/paths.ts`;
const pathsString = `${generatedWarning} export const paths = ${stringifyObject(
  extendedPaths
)} as const \n export const writablePaths = ${stringifyObject(extendedWritablePaths)} as const`;
const prettyPathsString = prettier.format(pathsString, {
  parser: `typescript`,
});

fs.writeFileSync(pathsFileName, prettyPathsString);

const interfaces = `
import { WrappedFact } from "@irs/js-factgraph-scala";
// @ts-ignore: Union type is too complex for CI to represent
`;

const wrappedFacts = processFactsToDigestWrapper(rawFacts);
const wrappedFactsFileName = `./src/fact-dictionary/generated/wrappedFacts.ts`;
// eslint-disable-next-line max-len
const wrappedFactsString = `${generatedWarning}  ${interfaces} export const wrappedFacts: WrappedFact[] = ${stringifyObject(
  wrappedFacts
)} as any
// cast to any as the typescript compiler cannot check a literal type with over 1000 facts
// because we cast again at the top of the file, the export is type safe.
// the only unsafe thing would be if the generator built something that did not match the WrappedFact interface
`;
const prettyWrappedFactsString = prettier.format(wrappedFactsString, {
  parser: `typescript`,
});

fs.writeFileSync(wrappedFactsFileName, prettyWrappedFactsString);

const submissionBlockingFacts = rawFacts.filter((f) => f.BlockSubmissionOnTrue !== undefined).map((f) => f[`@path`]);

const submissionBlockingFactsFilename = `./src/fact-dictionary/generated/submissionBlockingFacts.ts`;
// eslint-disable-next-line max-len
const submissionBlockingFactsString = `${generatedWarning} import { AbsolutePath } from "../Path.js"; export const submissionBlockingFacts: AbsolutePath[] = ${stringifyObject(
  submissionBlockingFacts
)}`;
const prettySubmissionBlockingFactsString = prettier.format(submissionBlockingFactsString, {
  parser: `typescript`,
});

fs.writeFileSync(submissionBlockingFactsFilename, prettySubmissionBlockingFactsString);

//ExportToState
const exportToStateFacts = rawFacts
  .filter((f) => f.Export !== undefined && f.Export[`@stateSystems`] === `true`)
  .map((f) => f[`@path`]);
const exportToStateFactsFilename = `./src/fact-dictionary/generated/exportToStateFacts.ts`;
// eslint-disable-next-line max-len
const exportToStateFactsString = `${generatedWarning} export const exportToStateFacts = ${stringifyObject(
  exportToStateFacts
)} as const`;
const prettyExportToStateFactsString = prettier.format(exportToStateFactsString, {
  parser: `typescript`,
});

fs.writeFileSync(exportToStateFactsFilename, prettyExportToStateFactsString);
