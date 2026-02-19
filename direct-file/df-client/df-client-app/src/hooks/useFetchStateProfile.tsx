import { useCallback, useEffect, useMemo, useState } from 'react';
import { StateOrProvince } from '../types/StateOrProvince.js';
import { read } from './useApiHook.js';
import { StateProfile, StateProfileResponse } from '../types/StateProfile.js';
import { Path } from '../flow/Path.js';
import useUrlTranslator from './useUrlTranslator.js';
import { useTranslation } from 'react-i18next';
import { useFactGraph } from '../factgraph/FactGraphContext.js';

const INITIAL_ERROR = undefined;
const INITIAL_SUCCESS = false;
const INITIAL_SKIPPED = false;

export const FILING_STATE_FACT_PATH = `/filingStateOrProvince`;

export type FetchStateProfileHookResponse = {
  stateProfile: StateProfile | null;
  isFetching: boolean;
  fetchError: unknown;
  fetchSuccess: boolean;
  fetchSkipped: boolean;
};

/**
 * Retrieves the state profile associated with the user's currently loaded tax return fact graph
 */
const useFetchStateProfile = (): FetchStateProfileHookResponse => {
  const { factGraph } = useFactGraph();

  const filingStateCode: StateOrProvince | undefined = useMemo(() => {
    if (factGraph) {
      const filingStateOrProvince = factGraph.get(Path.concretePath(FILING_STATE_FACT_PATH, null));
      return filingStateOrProvince.hasValue && filingStateOrProvince.complete
        ? (filingStateOrProvince.get.getValue().toUpperCase() as StateOrProvince)
        : undefined;
    }

    return undefined;
  }, [factGraph]);

  return useFetchStateProfileByStateCode(filingStateCode);
};

/**
 * Retrieves the state profile associated with the provided state code
 * @param stateCode
 */
export const useFetchStateProfileByStateCode = (stateCode?: StateOrProvince) => {
  const [isFetching, setIsFetching] = useState<boolean>(false);
  const [fetchError, setFetchError] = useState<unknown>(INITIAL_ERROR);
  const [fetchSuccess, setFetchSuccess] = useState<boolean>(INITIAL_SUCCESS);
  const [fetchSkipped, setFetchSkipped] = useState<boolean>(INITIAL_SKIPPED);
  const [rawStateProfile, setRawStateProfile] = useState<StateProfile | null>(null);
  const [stateProfile, setStateProfile] = useState<StateProfile | null>(null);

  const { translateStateProfileUrls } = useUrlTranslator();
  const { i18n } = useTranslation(`translation`);
  const [lastRenderLanguage, setLastRenderLanguage] = useState<string>(i18n.language);

  if (i18n.resolvedLanguage !== undefined && i18n.resolvedLanguage !== lastRenderLanguage) {
    setLastRenderLanguage(i18n.resolvedLanguage);
    if (rawStateProfile !== null) {
      const stateProfileWithTranslatedUrls = translateStateProfileUrls(rawStateProfile);
      setStateProfile(stateProfileWithTranslatedUrls);
    }
  }

  const getStateProfile = useCallback(
    async (stateCode: StateOrProvince) => {
      const handleFetchErrors = (error: unknown) => {
        setIsFetching(false);
        setFetchSuccess(false);
        setFetchError(error);
      };

      try {
        setIsFetching(true);
        setFetchSuccess(INITIAL_SUCCESS);
        setFetchError(INITIAL_ERROR);
        read<StateProfileResponse>(
          `${import.meta.env.VITE_BACKEND_URL}v1/state-api/state-profile?stateCode=${stateCode}`
        )
          .then((stateProfileResponse) => {
            setRawStateProfile(stateProfileResponse.stateProfile);
            if (stateProfileResponse.stateProfile !== null) {
              const stateProfileWithTranslatedUrls = translateStateProfileUrls(stateProfileResponse.stateProfile);
              setStateProfile(stateProfileWithTranslatedUrls);
            } else {
              setStateProfile(stateProfileResponse.stateProfile);
            }
            setIsFetching(false);
            setFetchSuccess(true);
          })
          .catch((error) => handleFetchErrors(error));
      } catch (error) {
        handleFetchErrors(error);
      }
    },
    [translateStateProfileUrls]
  );

  // fetch state profile
  useEffect(() => {
    if (stateCode) {
      void getStateProfile(stateCode);
      setFetchSkipped(INITIAL_SKIPPED);
    } else {
      setFetchSkipped(true);
    }
  }, [getStateProfile, stateCode]);

  return {
    stateProfile,
    isFetching,
    fetchError,
    fetchSuccess,
    fetchSkipped,
  };
};

export default useFetchStateProfile;
