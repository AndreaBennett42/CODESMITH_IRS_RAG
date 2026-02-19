import { ZodAny, ZodFormattedError } from 'zod';
import { SystemAlertKey } from '../../../context/SystemAlertContext/SystemAlertContext.js';
import type { ReactErrorPayload } from '../../../utils/errorBoundary.js';

// when adding new CodedExceptions, make sure to also make a corresponding update to
// TelemetryRequestKey on the backend
export const CodedExceptions = {
  INITIALIZE_CHECKLIST: `INITIALIZE_CHECKLIST`,
  SET_CURRENT_TAX_RETURN: `SET_CURRENT_TAX_RETURN`,
  ERROR_BOUNDARY: `ERROR_BOUNDARY`,
  MAX_POLLS_FOR_MEF_RECEIPT: `MAX_POLLS_FOR_MEF_RECEIPT`,
  MAX_POLLS_FOR_MEF_STATUS: `MAX_POLLS_FOR_MEF_STATUS`,
  DATA_IMPORT_ABOUT_YOU_BASIC_PARSE_ERROR: `DATA_IMPORT_ABOUT_YOU_BASIC_PARSE_ERROR`,
  DATA_IMPORT_IP_PIN_PARSE_ERROR: `DATA_IMPORT_IP_PIN_PARSE_ERROR`,
  DATA_IMPORT_W2_RESPONSE_PARSE_ERROR: `DATA_IMPORT_W2_RESPONSE_PARSE_ERROR`,
  DATA_IMPORT_SINGLE_W2_PARSE_ERROR: `DATA_IMPORT_SINGLE_W2_PARSE_ERROR`,
  DATA_IMPORT_LOADED_W2: `DATA_IMPORT_LOADED_W2`,
  DATA_IMPORT_IMPORTED_W2: `DATA_IMPORT_IMPORTED_W2`,
  DATA_IMPORT_1099_INT_RESPONSE_PARSE_ERROR: `DATA_IMPORT_1099_INT_RESPONSE_PARSE_ERROR`,
  DATA_IMPORT_LOADED_1099_INT: `DATA_IMPORT_LOADED_1099_INT`,
  DATA_IMPORT_SINGLE_1099_INT_PARSE_ERROR: `DATA_IMPORT_SINGLE_1099_INT_PARSE_ERROR`,
  DATA_IMPORT_IMPORTED_1099_INT: `DATA_IMPORT_IMPORTED_1099_INT`,
  DATA_IMPORT_1095A_RESPONSE_PARSE_ERROR: `DATA_IMPORT_1095A_RESPONSE_PARSE_ERROR`,
  DATA_IMPORT_LOADED_1095_A: `DATA_IMPORT_LOADED_1095_A`,
  PRIOR_YEAR_RETURN_STATUS_FETCH: `PRIOR_YEAR_RETURN_STATUS_FETCH`,
} as const;

export type CodedException = (typeof CodedExceptions)[keyof typeof CodedExceptions] | `SYSTEM_ALERT_${SystemAlertKey}`;

type DataImportParseErrorPayload = { unparsedData: unknown; formattedErrors: ZodFormattedError<ZodAny> };
type DataImportParseSuccessPayload = { unparsedData: unknown };
type DataImportClickPayload = { parsedData: unknown };

type CodedExceptionPayloadConfig = {
  SET_CURRENT_TAX_RETURN: {
    taxReturnID: string;
  };
  ERROR_BOUNDARY: ReactErrorPayload;
  DATA_IMPORT_ABOUT_YOU_BASIC_PARSE_ERROR: DataImportParseErrorPayload;
  DATA_IMPORT_IP_PIN_PARSE_ERROR: DataImportParseErrorPayload;
  DATA_IMPORT_W2_RESPONSE_PARSE_ERROR: DataImportParseErrorPayload;
  DATA_IMPORT_1099_INT_RESPONSE_PARSE_ERROR: DataImportParseErrorPayload;
  DATA_IMPORT_LOADED_W2: DataImportParseSuccessPayload;
  DATA_IMPORT_LOADED_1099_INT: DataImportParseSuccessPayload;
  DATA_IMPORT_SINGLE_W2_PARSE_ERROR: DataImportParseErrorPayload;
  DATA_IMPORT_SINGLE_1099_INT_PARSE_ERROR: DataImportParseErrorPayload;
  DATA_IMPORT_IMPORTED_W2: DataImportClickPayload;
  DATA_IMPORT_IMPORTED_1099_INT: DataImportClickPayload;
  DATA_IMPORT_1095A_RESPONSE_PARSE_ERROR: DataImportParseErrorPayload;
  DATA_IMPORT_LOADED_1095_A: DataImportParseSuccessPayload;
};

export type GetCodedExceptionPayload<ExceptionType extends CodedException> =
  ExceptionType extends keyof CodedExceptionPayloadConfig ? CodedExceptionPayloadConfig[ExceptionType] : undefined;

export type CodedExceptionConfig<ExceptionType extends CodedException> =
  GetCodedExceptionPayload<ExceptionType> extends undefined
    ? {
        key: ExceptionType;
        error?: unknown;
      }
    : {
        key: ExceptionType;
        payload: GetCodedExceptionPayload<ExceptionType>;
        error?: unknown;
      };

export type AnyCodedExceptionConfig = CodedExceptionConfig<CodedException>;
