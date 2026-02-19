import { z } from 'zod';

export function successPayloadOrWaitingOrError<T extends z.ZodTypeAny>(successPayload: T) {
  return z.discriminatedUnion(`state`, [
    z.object({
      state: z.literal(`incomplete`),
      createdAt: z.string().nullable(),
    }),
    z.object({
      state: z.literal(`success`),
      payload: successPayload,
      createdAt: z.string().nullable(),
    }),
    z.object({
      state: z.literal(`error`),
      createdAt: z.string().nullable(),
      errorType: z
        .enum([`parse-error`, `received-null`])
        .optional()
        .transform((val) => (val ? val : null)),
    }),
  ]);
}
