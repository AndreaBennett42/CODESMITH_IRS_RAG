import { z } from 'zod';
import { MaybeCompletePayload } from '../dataImportProfileTypes.js';
import { v5 as uuidv5 } from 'uuid';

import { isNotNull } from '../../../../utils/isNotNull.js';
import { DataImportIPPinPayloadSchema, IPPinPayload } from './IPPinSchema.js';
import { convertAddress, DataImportW2sPayloadSchema, w2Schema, W2sPayload } from './W2Schema.js';
import { AboutYouBasicPayload, DataImportAboutYouBasicPayloadSchema } from './AboutYouBasicSchema.js';
import {
  DataImportInterestIncomePayloadSchema,
  InterestIncomePayload,
  interestIncomeSchema,
} from './InterestIncomeSchema.js';
import { F1095APayload, F1095APayloadSchema } from './F1095ASchema.js';

const DATA_IMPORT_NAMESPACE_UUID = `123e4567-e89b-12d3-a456-426614174000`;

export const DataImportRootResponseSchema = z.object({
  data: z.object({
    timeSinceCreation: z.number(),
    aboutYouBasic: z.unknown().transform((data): MaybeCompletePayload<AboutYouBasicPayload> => {
      const parseResult = DataImportAboutYouBasicPayloadSchema.nullable().safeParse(data);
      if (parseResult.success) {
        if (parseResult.data === null) {
          return {
            state: `error`,
            errorType: `received-null`,
          };
        }
        if (parseResult.data.state === `success`) {
          return { ...parseResult.data, payload: parseResult.data.payload };
        }
        return parseResult.data;
      } else {
        return {
          state: `error`,
          errorType: `parse-error`,
        };
      }
    }),
    ipPin: z.unknown().transform((data): MaybeCompletePayload<IPPinPayload> => {
      const parseResult = DataImportIPPinPayloadSchema.nullable().safeParse(data);
      if (parseResult.success) {
        if (parseResult.data === null) {
          return {
            state: `error`,
            errorType: `received-null`,
          };
        }
        if (parseResult.data.state === `success`) {
          const pin = parseResult.data.payload.pin;
          if (pin === null) {
            return {
              state: `success`,
              createdAt: parseResult.data.createdAt,
              payload: {
                hasIpPin: false,
              },
            };
          }
          return {
            state: `success`,
            createdAt: parseResult.data.createdAt,
            payload: {
              hasIpPin: true,
              pin: pin,
            },
          };
        }
        return parseResult.data;
      } else {
        return {
          state: `error`,
          errorType: `parse-error`,
        };
      }
    }),
    w2s: z.unknown().transform((data): MaybeCompletePayload<W2sPayload> => {
      const parseResult = DataImportW2sPayloadSchema.nullable().safeParse(data);
      if (parseResult.success) {
        if (parseResult.data === null) {
          return {
            state: `error`,
            errorType: `received-null`,
          };
        }
        if (parseResult.data.state === `success`) {
          const payload: W2sPayload = (parseResult.data.payload.w2s || [])
            .map((rawW2) => {
              const w2ParseResult = w2Schema.safeParse(rawW2);
              if (!w2ParseResult.success) {
                return null;
              }

              const w2 = w2ParseResult.data;
              const id = uuidv5(
                `${w2.ein}_${w2.controlNumber}_${w2.employersAddress.nameLine}`,
                DATA_IMPORT_NAMESPACE_UUID
              );
              return {
                ...w2,
                employeeAddress: convertAddress(w2.employeeAddress),
                employersAddress: convertAddress(w2.employersAddress),
                id,
                isStandard: true,
                isCorrected: false,
              };
            })
            .filter(isNotNull);
          return {
            state: `success`,
            createdAt: parseResult.data.createdAt,
            payload,
          };
        } else {
          return parseResult.data;
        }
      } else {
        // TODO - change this to be more similar to the rest.
        // in the short term, since we moved all the json mocks at least
        // some issues are caused by an out of date build.
        return {
          state: `error`,
          errorType: `parse-error`,
        };
      }
    }),
    [`f1099Ints`]: z.unknown().transform((data): MaybeCompletePayload<InterestIncomePayload> => {
      const parseResult = DataImportInterestIncomePayloadSchema.nullable().safeParse(data);
      if (parseResult.success) {
        if (parseResult.data === null) {
          return {
            state: `error`,
            errorType: `received-null`,
          };
        }
        if (parseResult.data.state === `success`) {
          const payload: InterestIncomePayload = (parseResult.data.payload || [])
            .map((rawInterestIncome, index) => {
              const interestIncomeParseResult = interestIncomeSchema.safeParse(rawInterestIncome);
              if (!interestIncomeParseResult.success) {
                return null;
              }
              const interestIncome = interestIncomeParseResult.data;
              const id = uuidv5(`${index}_${interestIncome.payerName}`, DATA_IMPORT_NAMESPACE_UUID);
              return {
                ...interestIncome,
                has1099: true,
                id,
              };
            })
            .filter(isNotNull);
          return {
            state: `success`,
            createdAt: parseResult.data.createdAt,
            payload,
          };
        } else {
          return parseResult.data;
        }
      } else {
        return {
          state: `error`,
          errorType: `parse-error`,
        };
      }
    }),
    [`f1095a`]: z.unknown().transform((data): MaybeCompletePayload<F1095APayload> => {
      const parseResult = F1095APayloadSchema.nullable().safeParse(data);
      if (parseResult.success) {
        if (parseResult.data === null) {
          return {
            state: `error`,
            errorType: `received-null`,
          };
        }
        if (parseResult.data.state === `success`) {
          if (parseResult.data.payload !== null) {
            return {
              state: `success`,
              createdAt: parseResult.data.createdAt,
              payload: parseResult.data.payload,
            };
          }
        }
        return {
          state: `incomplete`,
          // createdAt: parseResult.data.createdAt,
          // payload: parseResult.data.payload,
        };
        // return parseResult.data;
      } else {
        return {
          state: `error`,
          errorType: `parse-error`,
        };
      }
    }),
  }),
});

export type DataImportRootResponseSchemaType = z.infer<typeof DataImportRootResponseSchema>;
