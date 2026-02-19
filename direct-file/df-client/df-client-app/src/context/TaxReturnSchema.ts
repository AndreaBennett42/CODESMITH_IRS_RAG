import { z } from 'zod';
import { TaxReturn } from '../types/core.js';

const taxReturnSubmissionSchema = z.object({
  id: z.string(),
  submitUserId: z.string(),
  createdAt: z.string(),
  receiptId: z.string().nullable(),
  submissionReceivedAt: z.string().nullable(),
});

const factValueSchema = z.object({
  $type: z.string(),
  item: z.unknown().refine((x) => x !== undefined, `Fact value item is required`),
});

const taxReturnSchema = z.object({
  id: z.string(),
  createdAt: z.string(),
  taxYear: z.number(),
  facts: z.record(factValueSchema),
  taxReturnSubmissions: z.array(taxReturnSubmissionSchema),
  isEditable: z.boolean(),
  surveyOptIn: z.boolean().nullable(),
});

export function maybeGetTaxReturnFromUnknown(input: unknown): TaxReturn | null {
  const result = taxReturnSchema.safeParse(input);
  if (result.success) {
    const { data } = result;
    return data;
  }

  return null;
}
