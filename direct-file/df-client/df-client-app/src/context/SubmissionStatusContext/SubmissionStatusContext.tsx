import { TaxReturnSubmissionStatus } from '../../types/core.js';
import {
  createContext,
  Dispatch,
  ReactNode,
  SetStateAction,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react';
import { read } from '../../hooks/useApiHook.js';
import { getCurrentTaxYearReturn, hasBeenSubmitted } from '../../utils/taxReturnUtils.js';
import { TaxReturnsContext } from '../TaxReturnsContext.js';

export type SubmissionStatusContextType = {
  submissionStatus: TaxReturnSubmissionStatus | undefined;
  setSubmissionStatus: Dispatch<SetStateAction<TaxReturnSubmissionStatus | undefined>>;
  fetchSubmissionStatus: (taxReturnId: string) => void;
  isFetching: boolean;
  fetchSuccess: boolean;
  fetchError: unknown | undefined;
  lastFetchAttempt: Date | null;
};

export const SubmissionStatusContext = createContext<SubmissionStatusContextType>({} as SubmissionStatusContextType);

type SubmissionStatusContextProviderProps = {
  children: ReactNode;
};

const INITIAL_SUCCESS = false;
const INITIAL_ERROR = undefined;

export const SubmissionStatusContextProvider = ({ children }: SubmissionStatusContextProviderProps) => {
  const { taxReturns } = useContext(TaxReturnsContext);

  const [submissionStatus, setSubmissionStatus] = useState<TaxReturnSubmissionStatus>();
  const [isFetching, setIsFetching] = useState(false);
  const [fetchSuccess, setFetchSuccess] = useState(INITIAL_SUCCESS);
  const [fetchError, setFetchError] = useState<unknown>(INITIAL_ERROR);
  const [lastFetchAttempt, setLastFetchAttempt] = useState<Date | null>(null);
  const currentTaxReturn = getCurrentTaxYearReturn(taxReturns);
  const needToFetchStatus = currentTaxReturn && hasBeenSubmitted(currentTaxReturn) && !submissionStatus;
  const fetchSubmissionStatus = useCallback(
    (taxReturnId: string) => {
      const handleFetchErrors = (error: unknown) => {
        setIsFetching(false);
        setFetchSuccess(false);
        setFetchError(error);
      };

      try {
        setIsFetching(true);
        setFetchSuccess(INITIAL_SUCCESS);
        setFetchError(INITIAL_ERROR);
        read<TaxReturnSubmissionStatus>(`${import.meta.env.VITE_BACKEND_URL}v1/taxreturns/${taxReturnId}/status`)
          .then((statusResponse) => {
            setSubmissionStatus(statusResponse);
            setIsFetching(false);
            setFetchSuccess(true);
            setFetchError(undefined);
          })
          .catch((e) => handleFetchErrors(e));
      } catch (e) {
        handleFetchErrors(e);
      } finally {
        setLastFetchAttempt(new Date());
      }
    },
    [setSubmissionStatus, setIsFetching]
  );

  useEffect(() => {
    if (needToFetchStatus) {
      fetchSubmissionStatus(currentTaxReturn.id);
    }
  }, [currentTaxReturn, needToFetchStatus, fetchSubmissionStatus]);

  const value = {
    submissionStatus,
    setSubmissionStatus,
    fetchSubmissionStatus,
    isFetching,
    fetchSuccess,
    fetchError,
    lastFetchAttempt,
  };

  return <SubmissionStatusContext.Provider value={value}>{children}</SubmissionStatusContext.Provider>;
};
