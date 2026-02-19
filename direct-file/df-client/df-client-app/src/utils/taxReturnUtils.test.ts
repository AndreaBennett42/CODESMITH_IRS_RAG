import { TaxReturn } from '../types/core.js';
import { CURRENT_TAX_YEAR } from '../constants/taxConstants.js';
import { v4 as uuidv4 } from 'uuid';
import { getLatestSubmission } from './taxReturnUtils.js';

describe(`taxReturnUtils`, () => {
  describe(`getLatestSubmission()`, () => {
    it(`returns undefined when there are no submissions to retrieve`, () => {
      const taxReturn: TaxReturn = {
        id: uuidv4(),
        createdAt: new Date().toISOString(),
        taxYear: parseInt(CURRENT_TAX_YEAR),
        facts: {},
        taxReturnSubmissions: [],
        isEditable: true,
        surveyOptIn: null,
      };

      const latest = getLatestSubmission(taxReturn);

      expect(latest).toBeUndefined();
    });
    it(`gets the latest submission`, () => {
      const userId = uuidv4();
      const taxReturn: TaxReturn = {
        id: uuidv4(),
        createdAt: new Date().toISOString(),
        taxYear: parseInt(CURRENT_TAX_YEAR),
        facts: {},
        surveyOptIn: null,
        taxReturnSubmissions: [
          {
            id: uuidv4(),
            receiptId: `old`,
            createdAt: `${CURRENT_TAX_YEAR}-01-01`,
            submitUserId: userId,
            submissionReceivedAt: `${CURRENT_TAX_YEAR}-01-01`,
          },
          {
            id: uuidv4(),
            receiptId: `new`,
            createdAt: `${CURRENT_TAX_YEAR}-04-15`,
            submitUserId: userId,
            submissionReceivedAt: `${CURRENT_TAX_YEAR}-04-15`,
          },
          {
            id: uuidv4(),
            receiptId: `middle`,
            createdAt: `${CURRENT_TAX_YEAR}-02-28`,
            submitUserId: userId,
            submissionReceivedAt: `${CURRENT_TAX_YEAR}-02-28`,
          },
        ],
        isEditable: false,
      };

      const latest = getLatestSubmission(taxReturn);

      expect(latest).not.toBeUndefined();
      expect(latest?.receiptId).toEqual(`new`);
    });
  });
});
