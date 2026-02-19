import { ReactNode, createContext } from 'react';
import { HelmetProvider } from 'react-helmet-async';
import { CURRENT_PHASE, ProjectPhaseMap, FeatureFlagMap, ProjectPhases, TODAY, DATES } from '../constants.js';

/**
 * Switches between phases based on the user's current date.
 *
 * This will not refresh, if the user leaves their browser open overnight.
 *
 * Dates are hardcoded to 2024 for the pilot.
 */
const calculatePhaseByDate = (): ProjectPhases => {
  // it is IMPORTANT that these be in chronological order
  if (TODAY >= DATES.CLOSED) {
    return `closed`;
  } else if (TODAY >= DATES.AFTER_DEADLINE) {
    return `after_deadline`;
  } else if (TODAY >= DATES.CLOSING_SOON) {
    return `closing_soon`;
  } else {
    return `open`;
  }
};

/** See `CURRENT_PHASE` in `constants.js`. */
const current_phase: ProjectPhases = CURRENT_PHASE ?? calculatePhaseByDate();

/** Control UX based which phase of the DF pilot we are in. See `CURRENT_PHASE` in `constants.js`. */
export const PilotPhaseContext = createContext<FeatureFlagMap>(ProjectPhaseMap[current_phase]);

type ProvidersProps = { children: ReactNode | ReactNode[] };

/** Give access to pilot phase feature flags. Exported separately for testing. */
export const PhaseProvider = ({ children }: ProvidersProps) => (
  <PilotPhaseContext.Provider value={ProjectPhaseMap[current_phase]}>{children}</PilotPhaseContext.Provider>
);

/** Wrap all the React context providers in a single component for convenience. */
const Providers = ({ children }: ProvidersProps) => (
  <HelmetProvider>
    <PhaseProvider>{children}</PhaseProvider>
  </HelmetProvider>
);

export default Providers;
