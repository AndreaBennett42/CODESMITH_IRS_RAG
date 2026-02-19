import { createStringEnum } from '../utils/enumUtils.js';

// The values from StateApiErrorCode.java which we might want to use on the client.
// See StateApiErrorCode.java for the full list
export const StateApiErrorCode = createStringEnum([`E_TAX_RETURN_NOT_ACCEPTED_OR_PENDING`]);
