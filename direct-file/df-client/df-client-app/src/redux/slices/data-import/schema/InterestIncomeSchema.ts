import { z } from 'zod';
import { successPayloadOrWaitingOrError } from './schemaPayload.js';
import { transformImportedDollar } from '../import-transform/transformImportedDollar.js';
import { sanitizeStringWithLimit } from '../import-transform/sanitizeStringWithLimit.js';
import { transformImportedTin } from '../import-transform/transformImportedTin.js';

export const interestIncomeSchema = z.object({
  payerName: z
    .string()
    .transform((input) => sanitizeStringWithLimit(input, `/interestReports/*/payer`))
    .refine((name) => name.length > 0, { message: `Invalid payerName` }),
  payerTin: z
    .string()
    .transform((input) => transformImportedTin(input))
    .refine((ein) => ein.length > 0, { message: `Invalid payerTin` }),
  fatcaFilingRequirementInd: z.boolean().nullable(),
  box1: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .nullable(),
  box2: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .nullable(),
  box3: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .nullable(),
  box4: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .nullable(),
  box6: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .nullable(),
  box8: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .nullable(),
  box9: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .nullable(),
  box10: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .nullable(),
  box11: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .nullable(),
  box12: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .nullable(),
  box13: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .nullable(),
  box14: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .nullable(),
});

export type InterestIncomePayload = (z.infer<typeof interestIncomeSchema> & {
  id: string;
})[];

export const DataImportInterestIncomePayloadSchema = successPayloadOrWaitingOrError(z.array(z.unknown()).nullable());
