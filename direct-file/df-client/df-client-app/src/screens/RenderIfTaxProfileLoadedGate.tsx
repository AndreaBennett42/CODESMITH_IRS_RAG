import { FC } from 'react';
import { FactGraphContextProvider } from '../factgraph/FactGraphContext.js';
import { useTaxProfileLoader } from './useTaxProfileLoader.js';

export interface RenderIfTaxProfileLoadedGateProps {
  children: React.ReactNode | React.ReactNode[];
}
export const RenderIfTaxProfileLoadedGate: FC<RenderIfTaxProfileLoadedGateProps> = ({ children }) => {
  const { currentTaxReturn, taxProfileNotYetAvailable, submissionStatus } = useTaxProfileLoader();

  if (taxProfileNotYetAvailable) {
    return null;
  }
  return (
    <FactGraphContextProvider
      existingFacts={currentTaxReturn.facts}
      taxReturnSubmissions={currentTaxReturn.taxReturnSubmissions}
      submissionStatus={submissionStatus}
    >
      {children}
    </FactGraphContextProvider>
  );
};
