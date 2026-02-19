import type { DataImportRootResponseSchemaType } from './schema/DataImportServiceResponse.js';
import type { DataImportProfile, DataImportRolloutBehavior } from './dataImportProfileTypes.js';
import { getDataImportMode } from '../../../constants/pageConstants.js';
import { assertNever } from 'assert-never';

const RETRY_TIMEOUT = 18000; // 18 seconds

// TODO - update input to use budget time from server.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function getIsOverTimeBudget(input: DataImportRootResponseSchemaType): boolean {
  const dataImportMode = getDataImportMode();
  if (dataImportMode === `clientside-intercept` || dataImportMode === `disabled`) {
    return true;
  }
  const { timeSinceCreation } = input.data;
  return timeSinceCreation > RETRY_TIMEOUT;
}

/*
 * Processing by payload allows one type of data to fail and not take down the whole import payload.
 */
export function processPopulateResult(
  input: DataImportRootResponseSchemaType,
  behavior: DataImportRolloutBehavior
): DataImportProfile {
  const isOverTimeBudget = getIsOverTimeBudget(input);
  const { aboutYouBasic, ipPin, w2s } = input.data;
  if (behavior === `DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2`) {
    return {
      data: {
        aboutYouBasic,
        ipPin,
        w2s,
        interestIncome: {
          state: `error`,
          errorType: `disabled`,
        },
        f1095A: {
          state: `error`,
          errorType: `disabled`,
        },
      },
      isOverTimeBudget,
    };
  }

  if (behavior === `DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2_PLUS_1099_INT`) {
    const interestIncome = input.data.f1099Ints;
    return {
      data: {
        aboutYouBasic,
        ipPin,
        w2s,
        interestIncome,
        f1095A: {
          state: `error`,
          errorType: `disabled`,
        },
      },
      isOverTimeBudget,
    };
  }

  if (behavior === `DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2_PLUS_1099_INT_PLUS_1095_A`) {
    const interestIncome = input.data.f1099Ints;
    const f1095A = input.data.f1095a;
    return {
      data: {
        aboutYouBasic,
        ipPin,
        w2s,
        interestIncome,
        f1095A,
      },
      isOverTimeBudget,
    };
  }

  if (behavior === `DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN`) {
    return {
      data: {
        aboutYouBasic,
        ipPin,
        w2s: {
          state: `error`,
          errorType: `disabled`,
        },
        interestIncome: {
          state: `error`,
          errorType: `disabled`,
        },
        f1095A: {
          state: `error`,
          errorType: `disabled`,
        },
      },
      isOverTimeBudget,
    };
  }

  if (behavior === `DATA_IMPORT_ABOUT_YOU_BASIC`) {
    return {
      data: {
        aboutYouBasic,
        ipPin: {
          state: `error`,
          errorType: `disabled`,
        },
        w2s: {
          state: `error`,
          errorType: `disabled`,
        },
        interestIncome: {
          state: `error`,
          errorType: `disabled`,
        },
        f1095A: {
          state: `error`,
          errorType: `disabled`,
        },
      },
      isOverTimeBudget,
    };
  }
  assertNever(behavior);
}
