import { z } from 'zod';
import { successPayloadOrWaitingOrError } from './schemaPayload.js';

const IPPinSchema = z.object({
  pin: z.string().length(6).nullable(),
});

export const DataImportIPPinPayloadSchema = successPayloadOrWaitingOrError(IPPinSchema);
export type IPPinPayload =
  | {
      hasIpPin: true;
      pin: string;
    }
  | {
      hasIpPin: false;
    };
