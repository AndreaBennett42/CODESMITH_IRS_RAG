import { Path } from '../../fact-dictionary/Path.js';
import { buildControlErrorId, buildFormControlId, buildHintId, buildHintKey, buildReadonlyHintKey } from './helpers.js';
import { ConcretePath } from '@irs/js-factgraph-scala';

const testConcretePath = `/test/concretePath` as ConcretePath;
const testSubfield = `testSubfield`;
const testControlId = buildFormControlId(testConcretePath);
const testPath = `/test/path` as Path;

describe(`helpers`, () => {
  it(`buildControlId with parameters: concretePath`, () => {
    expect(buildFormControlId(testConcretePath)).toEqual(`id-${testConcretePath}`);
  });

  it(`buildControlId with parameters: concretePath and subfield`, () => {
    expect(buildFormControlId(testConcretePath, testSubfield)).toEqual(`id-${testConcretePath}__${testSubfield}`);
  });

  it(`buildControlErrorId with parameters: controlId`, () => {
    expect(buildControlErrorId(testControlId)).toEqual(`id-${testConcretePath}__error-msg`);
  });

  it(`buildControlErrorId with parameters: controlId and subfield`, () => {
    expect(buildControlErrorId(testControlId, testSubfield)).toEqual(
      `id-${testConcretePath}__${testSubfield}__error-msg`
    );
  });

  it(`buildHintId with parameters: controlId`, () => {
    expect(buildHintId(testControlId)).toEqual(`${testControlId}__hint`);
  });

  it(`buildHintId with parameters: controlId and subfield`, () => {
    expect(buildHintId(testControlId, testSubfield)).toEqual(`${testControlId}__${testSubfield}__hint`);
  });

  it(`buildHintKey with parameters: path`, () => {
    expect(buildHintKey(testPath)).toEqual(`info./info${testPath}.helpText.hint`);
  });

  it(`buildHintKey with parameters: path and subfield`, () => {
    expect(buildHintKey(testPath, testSubfield)).toEqual(`info./info${testPath}.${testSubfield}.helpText.hint`);
  });

  it(`buildHintKey with parameters: path`, () => {
    expect(buildHintKey(undefined)).toEqual(``);
  });

  it(`buildHintKey with parameters: path and subfield`, () => {
    expect(buildHintKey(undefined, testSubfield)).toEqual(``);
  });
  it(`buildReadonlyHintKey with parameters: path`, () => {
    expect(buildReadonlyHintKey(testPath)).toEqual(`info.${testPath}.readOnlyField`);
  });
  it(`buildReadonlyHintKey with parameters: path`, () => {
    expect(buildReadonlyHintKey(undefined)).toEqual(``);
  });
});
