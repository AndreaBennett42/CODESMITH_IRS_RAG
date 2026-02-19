import type { CodedException, GetCodedExceptionPayload } from './codedException.js';

export type ProcessedError = {
  message: string | null;
  stack: string | null;
  name: string | null;
  code: string | null;
  hasCause: boolean;
} | null;

export type BoundStoredLoggedError<T extends CodedException> = {
  id: string;
  key: T;
  timestampSeconds: string;
  error: ProcessedError;
  payload: GetCodedExceptionPayload<T>;
};

export type StoredLoggedError = BoundStoredLoggedError<CodedException>;
