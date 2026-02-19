import { createContext, useContext } from 'react';

export const FactGraphTranslationContext = createContext<{ shouldFetchTranslationValuesFromFactGraph: boolean }>({
  shouldFetchTranslationValuesFromFactGraph: true,
});

export const useFactGraphTranslationContext = () => {
  return useContext(FactGraphTranslationContext);
};
