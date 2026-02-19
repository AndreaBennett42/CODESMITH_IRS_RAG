import { areAnyRejectionsNotFixable } from './submissionStatusUtils.js';
import { RejectedStatus } from '../types/core.js';
import { MEF_REJECTION_ERROR_CODES } from '../constants/rejectionConstants.js';

describe(`submissionStatusUtils`, () => {
  describe(areAnyRejectionsNotFixable.name, () => {
    it(`returns true if even one rejection code is not fixable`, () => {
      const rejectionCodes: RejectedStatus[] = [
        {
          MeFErrorCode: `IND-181-01`,
          MeFDescription: `not used`,
          TranslationKey: `not used`,
        },
        {
          MeFErrorCode: MEF_REJECTION_ERROR_CODES.UNFIXABLE_BY_DF[0],
          MeFDescription: `not used`,
          TranslationKey: `not used`,
        },
      ];

      expect(areAnyRejectionsNotFixable(rejectionCodes)).toBeTruthy();
    });

    it(`returns false if all of the rejection codes are fixable`, () => {
      const rejectionCodes: RejectedStatus[] = [
        {
          MeFErrorCode: `IND-181-01`,
          MeFDescription: `not used`,
          TranslationKey: `not used`,
        },
        {
          MeFErrorCode: `Some fixable error code`,
          MeFDescription: `not used`,
          TranslationKey: `not used`,
        },
        {
          MeFErrorCode: `Some other fixable error code`,
          MeFDescription: `not used`,
          TranslationKey: `not used`,
        },
      ];

      expect(areAnyRejectionsNotFixable(rejectionCodes)).toBeFalsy();
    });
  });
});
