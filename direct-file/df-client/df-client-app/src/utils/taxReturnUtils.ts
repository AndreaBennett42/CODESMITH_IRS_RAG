import { TaxReturn } from '../types/core.js';
import { CURRENT_TAX_YEAR } from '../constants/taxConstants.js';

export const anyHasStarted = (taxReturns: TaxReturn[]) => {
  return taxReturns.some((tr: TaxReturn) => {
    return tr.id;
  });
};

export const hasBeenSubmitted = (taxReturn: TaxReturn) => taxReturn.taxReturnSubmissions.length > 0;

export const anyHasBeenSubmitted = (taxReturns: TaxReturn[]) => {
  return taxReturns.some((tr: TaxReturn) => {
    return hasBeenSubmitted(tr);
  });
};

export const getLatestSubmission = (taxReturn: TaxReturn) => {
  return taxReturn.taxReturnSubmissions
    .slice()
    .sort((s1, s2) => new Date(s2.createdAt).getTime() - new Date(s1.createdAt).getTime())
    .at(0);
};

export const getTaxReturnById = (taxReturns: TaxReturn[], taxId: string | null) =>
  taxId ? taxReturns.find((tr) => tr.id === taxId) : undefined;

export const getCurrentTaxYearReturn = (taxReturns: TaxReturn[]) =>
  taxReturns.find((tr) => tr.taxYear.toString() === CURRENT_TAX_YEAR);
