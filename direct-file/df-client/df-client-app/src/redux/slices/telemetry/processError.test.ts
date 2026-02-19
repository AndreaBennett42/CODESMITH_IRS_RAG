// Assuming the module file is named 'errorProcessor.ts'
import { processError } from './processError.js';
import { ProcessedError } from './types.js';

describe(`processError`, () => {
  test(`should return null when inputError is null`, () => {
    expect(processError(null)).toBeNull();
  });

  test(`should return null when inputError is undefined`, () => {
    expect(processError(undefined)).toBeNull();
  });

  test(`should return null when inputError is a string`, () => {
    expect(processError(`Some error`)).toBeNull();
  });

  test(`should return null when inputError is a number`, () => {
    expect(processError(404)).toBeNull();
  });

  test(`should return null when inputError is a boolean`, () => {
    expect(processError(false)).toBeNull();
  });

  test(`should process error object with string properties correctly`, () => {
    const inputError = {
      name: `TypeError`,
      message: `An unexpected error occurred.`,
      stack: `Error stack trace`,
      code: `ERR_TYPE`,
      cause: `Invalid type`,
    };

    const expectedOutput: ProcessedError = {
      name: `TypeError`,
      message: `An unexpected error occurred.`,
      stack: `Error stack trace`,
      code: `ERR_TYPE`,
      hasCause: true,
    };

    expect(processError(inputError)).toEqual(expectedOutput);
  });

  test(`should convert number and bigint properties to strings`, () => {
    const inputError = {
      name: 123,
      message: 456n,
      stack: 789,
      code: 0n,
    };

    const expectedOutput: ProcessedError = {
      name: `123`,
      message: `456`,
      stack: `789`,
      code: `0`,
      hasCause: false,
    };

    expect(processError(inputError)).toEqual(expectedOutput);
  });

  test(`should return null for non-stringifiable properties`, () => {
    const inputError = {
      name: Symbol(`errorName`),
      message: { info: `Some info` },
      stack: [1, 2, 3],
      code: () => {},
    };

    const expectedOutput: ProcessedError = {
      name: null,
      message: null,
      stack: null,
      code: null,
      hasCause: false,
    };

    expect(processError(inputError)).toEqual(expectedOutput);
  });

  test(`should set hasCause to true when cause property is truthy`, () => {
    const inputError = { cause: `Some cause` };

    const result = processError(inputError);
    expect(result?.hasCause).toBe(true);
  });

  test(`should set hasCause to false when cause property is falsy or not present`, () => {
    const inputError1 = { cause: null };
    const inputError2 = {};

    expect(processError(inputError1)?.hasCause).toBe(false);
    expect(processError(inputError2)?.hasCause).toBe(false);
  });

  test(`should handle empty object input`, () => {
    const expectedOutput: ProcessedError = {
      name: null,
      message: null,
      stack: null,
      code: null,
      hasCause: false,
    };

    expect(processError({})).toEqual(expectedOutput);
  });
});
