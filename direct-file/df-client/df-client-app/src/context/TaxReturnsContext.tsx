import { ReactNode, createContext } from 'react';
import { TaxReturn } from '../types/core.js';

import { useAppSelector, useAppStore } from '../redux/hooks.js';
import { fetchTaxReturns } from '../redux/slices/tax-return/taxReturnSlice.js';
import { useFetchTaxReturnsIfNeeded } from '../hooks/useInitializeFetchRequiredData.js';

export type TaxReturnsContextType = {
  currentTaxReturnId: string | null;
  taxReturns: TaxReturn[];

  // Can be used to manually refresh tax returns
  fetchTaxReturns: () => void;
  isFetching: boolean;
  fetchSuccess: boolean;
};

export const TaxReturnsContext = createContext<TaxReturnsContextType>({} as TaxReturnsContextType);

type TaxReturnsContextProviderProps = {
  children: ReactNode;
};

export const TaxReturnsContextProvider = ({ children }: TaxReturnsContextProviderProps) => {
  const { taxReturns, isFetching, fetchSuccess } = useFetchTaxReturnsIfNeeded();
  const store = useAppStore();
  const currentTaxReturnId = useAppSelector((state) => state.taxReturns.data.currentTaxReturnId);

  return (
    <TaxReturnsContext.Provider
      value={{
        currentTaxReturnId,
        taxReturns,
        fetchTaxReturns: () => store.dispatch(fetchTaxReturns()),
        isFetching,
        fetchSuccess,
      }}
    >
      {children}
    </TaxReturnsContext.Provider>
  );
};
