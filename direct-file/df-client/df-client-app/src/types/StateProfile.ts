import { StateOrProvince } from './StateOrProvince.js';

// Align with state-tax StateProfile
export type StateProfile = {
  stateCode: StateOrProvince;
  landingUrl: string;
  defaultRedirectUrl: string | null;
  departmentOfRevenueUrl: string | null;
  filingRequirementsUrl: string | null;
  transferCancelUrl: string | null;
  waitingForAcceptanceCancelUrl: string | null;
  redirectUrls: string[];
  languages: Record<string, string>;
  taxSystemName: string;
  acceptedOnly: boolean;
  // customFilingDeadline is timezone agnostic.
  // e.g. A value of 04-17 23:59:59.999999 is 04-17 23:59:59.999999 regardless of the user's timezone
  customFilingDeadline: string | null;
};

export type StateProfileResponse = {
  stateProfile: StateProfile | null;
  error: string | null;
};
