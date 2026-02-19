import { z } from 'zod';
import { transformImportedDateWithLimit } from '../import-transform/transformImportedDateWithLimit.js';
import { transformImportedFullAddress } from '../import-transform/transformImportedFullAddress.js';
import { transformImportedStringWithLimit } from '../import-transform/transformImportedStringWithLimit.js';
import { transformImportedPhone } from '../import-transform/transformImportedPhone.js';
import { transformImportedStateOrProvence } from '../import-transform/transformImportedStateOrProvence.js';
import { transformImportedCity } from '../import-transform/transformImportedCity.js';
import { transformImportedAddress } from '../import-transform/transformImportedAddress.js';
import { successPayloadOrWaitingOrError } from './schemaPayload.js';
import { supportedStates } from './schemaConstants.js';

export const AboutYouBasicAddressSchema = z.object({
  mailingAddress: z.string().optional().transform(transformImportedAddress).nullable(),
  streetAddress: z.string().transform(transformImportedAddress),
  streetAddressLine2: z.string().transform(transformImportedAddress).nullable(),
  city: z.string().transform(transformImportedCity),
  postalCode: z.string(),
  stateOrProvence: z.enum(supportedStates).transform(transformImportedStateOrProvence),
});

export type AboutYouBasicAddress = z.infer<typeof AboutYouBasicAddressSchema>;

const AboutYouBasicSchema = z
  .object({
    source: z.literal(`SADI`),
    tags: z.array(z.string()),
    createdDate: z.string().date(),
    email: z.string(),
    firstName: z.string().transform((input) => transformImportedStringWithLimit(input, `/filers/*/firstName`)),
    lastName: z.string().transform((input) => transformImportedStringWithLimit(input, `/filers/*/lastName`)),
    middleInitial: z
      .string()
      .max(1)
      .transform((input) => transformImportedStringWithLimit(input, `/filers/*/writableMiddleInitial`))
      .nullable(),
    dateOfBirth: z
      .string()
      .date()
      .transform((input) => transformImportedDateWithLimit(input, `/filers/*/dateOfBirth`)),
    landlineNumber: z.string().nullable().transform(transformImportedPhone),
    mobileNumber: z.string().nullable().transform(transformImportedPhone),
    ...AboutYouBasicAddressSchema.shape,
  })
  .transform((input) => {
    const { mailingAddress, streetAddress, streetAddressLine2, city, postalCode, stateOrProvence } = input;
    return {
      ...input,
      ...transformImportedFullAddress({
        mailingAddress,
        streetAddress,
        streetAddressLine2,
        city,
        postalCode,
        stateOrProvence,
      }),
    };
  });

export const DataImportAboutYouBasicPayloadSchema = successPayloadOrWaitingOrError(AboutYouBasicSchema);

export type AboutYouBasicPayload = z.infer<typeof AboutYouBasicSchema>;
