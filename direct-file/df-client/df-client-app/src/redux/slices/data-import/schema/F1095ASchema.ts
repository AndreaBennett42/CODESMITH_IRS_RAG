import { z } from 'zod';
import { successPayloadOrWaitingOrError } from './schemaPayload.js';

export const F1095ASchema = z.object({
  has1095A: z.boolean().nullable(),
});

export type F1095APayload = z.infer<typeof F1095ASchema> & {
  id: string;
};

export const F1095APayloadSchema = successPayloadOrWaitingOrError(z.any().nullable());
