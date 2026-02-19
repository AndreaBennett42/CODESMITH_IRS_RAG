import { Address } from '@irs/js-factgraph-scala';
import type { AboutYouBasicPayload } from './schema/AboutYouBasicSchema.js';
import type { IPPinPayload } from './schema/IPPinSchema.js';
import type { W2sPayload } from './schema/W2Schema.js';
import { InterestIncomePayload } from './schema/InterestIncomeSchema.js';
import { F1095APayload } from './schema/F1095ASchema.js';

export type MaybeCompletePayload<T> =
  | {
      readonly state: `incomplete`;
    }
  | {
      readonly state: `error`;
      readonly errorType: `parse-error` | `received-null` | `disabled` | null;
    }
  | {
      readonly state: `success`;
      readonly createdAt: string | null;
      readonly payload: T;
    };

export type DataImportProfile = {
  data: {
    aboutYouBasic: MaybeCompletePayload<AboutYouBasicPayload>;
    ipPin: MaybeCompletePayload<IPPinPayload>;
    w2s: MaybeCompletePayload<W2sPayload>;
    interestIncome: MaybeCompletePayload<InterestIncomePayload>;
    f1095A: MaybeCompletePayload<F1095APayload>;
  };
  isOverTimeBudget: boolean;
};

export type ImportedAddress = Address & { nameLine: string; nameLine2: string | null };

export type DataImportRolloutBehavior =
  | `DATA_IMPORT_ABOUT_YOU_BASIC`
  | `DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN`
  | `DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2`
  | `DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2_PLUS_1099_INT`
  | `DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2_PLUS_1099_INT_PLUS_1095_A`;
