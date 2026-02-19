import { useContext, createContext } from 'react';

export const FilterContentContext = createContext<{ shouldFilterContentBasedOnTaxState: boolean }>({
  shouldFilterContentBasedOnTaxState: true,
});

export const useFilterContentContext = () => {
  return useContext(FilterContentContext);
};
