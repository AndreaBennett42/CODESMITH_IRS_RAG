import { useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../redux/hooks.js';
import { fetchTaxReturns } from '../redux/slices/tax-return/taxReturnSlice.js';

export const useFetchTaxReturnsIfNeeded = () => {
  const dispatch = useAppDispatch();
  const taxReturns = useAppSelector((state) => state.taxReturns.data.taxReturns);
  const isFetching = useAppSelector((state) => state.taxReturns.data.isFetching);
  const fetchSuccess = useAppSelector((state) => state.taxReturns.data.fetchSuccess);
  const hasFetchError = useAppSelector((state) => state.taxReturns.data.hasFetchError);
  const hasFinishedFetching = !isFetching && (fetchSuccess || hasFetchError);
  const needToLoadTaxReturns = taxReturns.length === 0 && !hasFinishedFetching;
  // Load tax returns if needed
  useEffect(() => {
    if (needToLoadTaxReturns) {
      dispatch(fetchTaxReturns());
    }
  }, [needToLoadTaxReturns, dispatch]);

  return {
    taxReturns,
    isFetching,
    fetchSuccess,
  };
};
