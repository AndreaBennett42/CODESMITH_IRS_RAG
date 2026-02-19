import { z } from 'zod';
import { transformImportedDollar } from '../import-transform/transformImportedDollar.js';
import { transformImportedEin } from '../import-transform/transformImportedEin.js';
import { transformImportedFullW2Address, isValidAddress } from '../import-transform/transformImportedFullAddress.js';
import { transformImportedStringWithLimit } from '../import-transform/transformImportedStringWithLimit.js';
import { transformImportedAddress } from '../import-transform/transformImportedAddress.js';
import { transformImportedCity } from '../import-transform/transformImportedCity.js';
import { transformImportedStateOrProvence } from '../import-transform/transformImportedStateOrProvence.js';
import { ImportedAddress } from '../dataImportProfileTypes.js';
import { successPayloadOrWaitingOrError } from './schemaPayload.js';
import { supportedStates } from './schemaConstants.js';
import { sanitizeStringWithLimit } from '../import-transform/sanitizeStringWithLimit.js';

export const W2AddressSchema = z.object({
  streetAddressLine1: z.string().min(1).transform(transformImportedAddress),
  streetAddressLine2: z.string().transform(transformImportedAddress).nullable(),
  city: z.string().min(3).max(22).transform(transformImportedCity),
  zipCode: z.string().min(1),
  state: z.enum(supportedStates).transform(transformImportedStateOrProvence),
});

export const DataImportW2sPayloadSchema = successPayloadOrWaitingOrError(
  z.object({
    w2s: z.array(z.unknown()).nullable(),
  })
);

export const w2Schema = z.object({
  controlNumber: z.string(),
  ein: z
    .string()
    .transform(transformImportedEin)
    .refine((ein) => ein.length > 0, { message: `Invalid EIN` }),
  employeeAddress: z
    .object({
      nameLine: z.string().min(1),
      nameLine2: z.string().optional().nullable(),
      ...W2AddressSchema.shape,
    })
    .transform((input) => {
      const { streetAddressLine1, streetAddressLine2, city, zipCode, state } = input;
      return {
        ...input,
        ...transformImportedFullW2Address({ streetAddressLine1, streetAddressLine2, city, zipCode, state }),
      };
    })
    .refine(isValidAddress, { message: `Invalid employeeAddress` }),
  employersAddress: z
    .object({
      nameLine: z
        .string()
        .transform((input) => sanitizeStringWithLimit(input, `/formW2s/*/employerName`))
        .refine((name) => name.length > 0, { message: `Invalid employerName` }),
      nameLine2: z
        .string()
        .transform((input) => transformImportedStringWithLimit(input, `/formW2s/*/writableEmployerNameLine2`))
        .optional()
        .nullable(),
      ...W2AddressSchema.shape,
    })
    .transform((input) => {
      const { streetAddressLine1, streetAddressLine2, city, zipCode, state } = input;
      return {
        ...input,
        ...transformImportedFullW2Address({ streetAddressLine1, streetAddressLine2, city, zipCode, state }),
      };
    })
    .refine(isValidAddress, { message: `Invalid employersAddress` }),
  wagesTipsOtherCompensation: z
    .string()
    .transform((input) => transformImportedDollar(input))
    .refine((wages) => wages.length > 0, { message: `Invalid wagesTipsOtherCompensation` }),
  federalIncomeTaxWithheld: z.string().transform((input) => transformImportedDollar(input)),
  socialSecurityWages: z.string().transform((input) => transformImportedDollar(input)),
  socialSecurityTaxWithheld: z.string().transform((input) => transformImportedDollar(input)),
  medicareWagesAndTips: z.string().transform((input) => transformImportedDollar(input)),
  medicareTaxWithheld: z.string().transform((input) => transformImportedDollar(input)),
  socialSecurityTips: z.string().transform((input) => transformImportedDollar(input)),
  allocatedTips: z.string().transform((input) => transformImportedDollar(input)),
  dependentCareBenefits: z.string().transform((input) => transformImportedDollar(input)),
  nonQualifiedPlans: z.string().transform((input) => transformImportedDollar(input)),
  statutoryEmployeeIndicator: z.boolean(),
  thirdPartySickPayIndicator: z.boolean(),
  retirementPlanIndicator: z.boolean(),
});

export const schemaWithoutAddresses = w2Schema.omit({ employersAddress: true, employeeAddress: true });

export type W2sPayload = (z.infer<typeof schemaWithoutAddresses> & {
  id: string;
  isStandard: boolean;
  isCorrected: boolean;
  employersAddress: ImportedAddress | null;
  employeeAddress: ImportedAddress | null;
})[];

export function convertAddress(
  input: {
    state: string;
    streetAddressLine2: string | null;
    city: string;
    nameLine: string;
    nameLine2?: string | undefined | null;
    streetAddressLine1: string;
    zipCode: string;
  } | null
): ImportedAddress | null {
  if (input === null) {
    return null;
  }
  const { state, streetAddressLine2, city, nameLine, nameLine2, streetAddressLine1, zipCode } = input;

  return {
    nameLine,
    nameLine2: nameLine2 || ``,
    streetAddress: streetAddressLine1,
    streetAddressLine2: streetAddressLine2 || ``,
    city,
    stateOrProvence: state,
    postalCode: zipCode,
    country: `USA`,
  };
}

export type W2Address = z.infer<typeof W2AddressSchema>;
