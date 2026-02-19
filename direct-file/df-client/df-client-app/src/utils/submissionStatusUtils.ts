import { MEF_REJECTION_ERROR_CODES } from '../constants/rejectionConstants.js';
import { RejectedStatus } from '../types/core.js';

export const areAnyRejectionsNotFixable = (rejectionCodes: RejectedStatus[]) => {
  return rejectionCodes.some((code) => MEF_REJECTION_ERROR_CODES.UNFIXABLE_BY_DF.includes(code.MeFErrorCode));
};
