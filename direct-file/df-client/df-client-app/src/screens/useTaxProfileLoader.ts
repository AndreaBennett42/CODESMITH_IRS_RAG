import { useContext } from 'react';
import { SubmissionStatusContext } from '../context/SubmissionStatusContext/SubmissionStatusContext.js';
import { useAppSelector } from '../redux/hooks.js';
import { selectCurrentTaxReturn } from '../redux/slices/tax-return/taxReturnSlice.js';
import { selectDataImportFetchState } from '../redux/slices/data-import/dataImportProfileSlice.js';
import { TaxReturn, TaxReturnSubmissionStatus } from '../types/core.js';

// This type allows callers to check `taxProfileNotYetAvailable` to know if
// the currentTaxReturn is available in a type safe way.
type TaxProfileLoaderReturn = {
  taxReturnFetchCompleted: boolean;
  submissionStatus: TaxReturnSubmissionStatus | undefined;
} & (
  | {
      currentTaxReturn: TaxReturn;
      taxProfileNotYetAvailable: false;
    }
  | {
      currentTaxReturn: TaxReturn | undefined;
      taxProfileNotYetAvailable: true;
    }
);

export function useTaxProfileLoader(): TaxProfileLoaderReturn {
  const {
    submissionStatus,
    fetchSuccess: statusFetchSuccess,
    fetchError: statusFetchError,
  } = useContext(SubmissionStatusContext);

  const currentTaxReturn = useAppSelector(selectCurrentTaxReturn);
  const taxReturns = useAppSelector((store) => store.taxReturns.data.taxReturns);

  const isFetchingTaxReturns = useAppSelector((store) => store.taxReturns.data.isFetching);
  const taxReturnsFetchSuccess = useAppSelector((store) => store.taxReturns.data.fetchSuccess);
  const taxReturnsFetchError = useAppSelector((store) => store.taxReturns.data.hasFetchError);
  const dataImportFetchStatus = useAppSelector(selectDataImportFetchState);

  const taxReturnFetchCompleted = !isFetchingTaxReturns && (taxReturnsFetchSuccess || taxReturnsFetchError);
  const dataImportFetchCompleted = dataImportFetchStatus === `settled`;

  const shouldWaitForStatus = currentTaxReturn && currentTaxReturn.taxReturnSubmissions?.length > 0;
  const statusFetchCompleted = statusFetchSuccess || statusFetchError;

  const taxProfileNotYetAvailable =
    taxReturns.length === 0 ||
    currentTaxReturn?.facts === undefined ||
    !dataImportFetchCompleted ||
    (!!shouldWaitForStatus && !statusFetchCompleted);

  if (taxProfileNotYetAvailable) {
    return {
      taxReturnFetchCompleted,
      currentTaxReturn,
      taxProfileNotYetAvailable,
      submissionStatus,
    };
  }

  return {
    taxReturnFetchCompleted,
    currentTaxReturn,
    taxProfileNotYetAvailable,
    submissionStatus,
  };
}
