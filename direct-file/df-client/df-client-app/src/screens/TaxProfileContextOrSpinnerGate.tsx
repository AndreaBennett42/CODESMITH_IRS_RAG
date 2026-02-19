import { useEffect, FC } from 'react';
import { FactGraphContextProvider } from '../factgraph/FactGraphContext.js';
import { useNavigate } from 'react-router-dom';
import { useTaxProfileLoader } from './useTaxProfileLoader.js';
import LoadingIndicator from '../components/LoadingIndicator/LoadingIndicator.js';

export interface TaxProfileContextOrSpinnerGateProps {
  children: React.ReactNode | React.ReactNode[];
}

const FALLBACK_LOCATION = `/home`;

/**
 * This parent component, without rendering any markup, should be applied to any
 * route that requires that the FactGraph be loaded with the user's current facts.
 * If facts are not already loaded for the current tax year, the component will load them.
 * Loading the fact graph has been delegated to the TaxReturnsContextProvider.
 */
export const TaxProfileContextOrSpinnerGate: FC<TaxProfileContextOrSpinnerGateProps> = ({ children }) => {
  const navigate = useNavigate();
  const { taxReturnFetchCompleted, currentTaxReturn, taxProfileNotYetAvailable, submissionStatus } =
    useTaxProfileLoader();

  useEffect(() => {
    // If we have attempted to fetch tax returns, success or not, send the user home if they do not have an active
    // tax return for which to instantiate the fact graph singleton, because they won't be able to get any further
    // wherever they're trying to go
    if (taxReturnFetchCompleted && currentTaxReturn === undefined) {
      navigate(FALLBACK_LOCATION);
    }
  }, [currentTaxReturn, navigate, taxReturnFetchCompleted]);

  if (taxProfileNotYetAvailable) {
    return <LoadingIndicator />;
  }
  // currentTaxReturn.facts needs strict equality to undefined, since it could be an empty object
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
