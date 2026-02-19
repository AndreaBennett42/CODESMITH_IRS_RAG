import { ProcessedError } from './types.js';

/*
 * Javascript errors _do_ have types in that `new Error('ohai')` has common properties
 * however, it is not required that an error type is thrown.
 * The following code is valid and is unfortunatley common in libraries for various reasons:
 *
 * try {
 *   throw {message: 'ohai'};
 * } catch (e) {
 *   console.log(e.message) // e is not an `Error` here but it still works
 * }
 *
 */
export function processError(inputError: unknown): ProcessedError {
  if (typeof inputError !== `object` || inputError === null) {
    return null;
  }

  const name = `name` in inputError ? maybeStringify(inputError.name) : null;
  const message = `message` in inputError ? maybeStringify(inputError.message) : null;
  const stack = `stack` in inputError ? maybeStringify(inputError.stack) : null;
  const code = `code` in inputError ? maybeStringify(inputError.code) : null;
  const hasCause = getHasCause(inputError);

  return {
    name,
    message,
    stack,
    code,
    hasCause,
  };
}

function getHasCause(inputError: object): boolean {
  if (!(`cause` in inputError)) {
    return false;
  }
  return !!inputError.cause;
}

function maybeStringify(input: unknown): string | null {
  if (typeof input === `string`) {
    return input;
  }

  if (typeof input === `number` || typeof input === `bigint`) {
    return input.toString();
  }

  return null;
}
