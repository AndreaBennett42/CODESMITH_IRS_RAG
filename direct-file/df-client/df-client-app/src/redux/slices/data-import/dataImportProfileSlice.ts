import { createSlice, createAsyncThunk, SerializedError } from '@reduxjs/toolkit';
import { DataImportRootResponseSchema } from './schema/DataImportServiceResponse.js';
import { getProfileForClientsideIntercept } from './DevDataImportConfig.js';
import {
  DEFAULT_DATA_IMPORT_BEHAVIOR,
  getDataImportMode,
  LOCAL_DATA_IMPORT_BEHAVIOR,
} from '../../../constants/pageConstants.js';
import { TaxReturn } from '../../../types/core.js';
import { fetchDataImportProfile } from './fetchDataImportProfile.js';
import { isDev } from '../../../env/envHelpers.js';
import { assertNever } from 'assert-never';
import { processPopulateResult } from './processPopulateResult.js';
import { DataImportProfile, DataImportRolloutBehavior } from './dataImportProfileTypes.js';
import { RootState } from '../../store.js';

function getFrontendRolloutBehavior(currentTaxReturn: TaxReturn): DataImportRolloutBehavior {
  const backendBehavior = currentTaxReturn.dataImportBehavior as DataImportRolloutBehavior;
  if (isDev()) {
    return LOCAL_DATA_IMPORT_BEHAVIOR;
  }
  if (!backendBehavior) {
    return DEFAULT_DATA_IMPORT_BEHAVIOR;
  }
  return backendBehavior;
}

// Thunk to fetch data from an API
export const fetchProfile = createAsyncThunk(
  `data-import/fetchProfile`,
  async (args: { currentTaxReturn: TaxReturn }, { rejectWithValue, dispatch, getState }) => {
    const dataImportMode = getDataImportMode();
    const rolloutBehavior = getFrontendRolloutBehavior(args.currentTaxReturn);
    const currentTaxReturnID = args.currentTaxReturn.id;
    if (dataImportMode === `real-api`) {
      try {
        const dataImportResult = await fetchDataImportProfile(currentTaxReturnID);
        const parseResult = DataImportRootResponseSchema.safeParse(dataImportResult);
        if (!parseResult.success) {
          return rejectWithValue(`Invalid response`);
        }

        const processedResult = processPopulateResult(parseResult.data, rolloutBehavior);
        const { isOverTimeBudget } = processedResult;

        const awaitingBackend = getFetchStatusForProfile(processedResult) === `waiting-for-backend`;
        const shouldRetry = !isOverTimeBudget && awaitingBackend;
        if (shouldRetry) {
          setTimeout(() => {
            // After resuming after the set timeout, we need to grab the current state so that we know
            // whether we're still having to refetch. If we don't need to refetch, don't trigger the
            // refetch. Ideally, we would only have one timeout in the system but given our usage
            // of useEffects in the codebase, I don't totally trust this not to happen with quick
            // mounting/unmounting of components.
            const state = getState() as RootState;
            const dataImportProfile = state.dataImportProfile.data;
            if (dataImportProfile.status === `complete`) {
              const { isOverTimeBudget } = dataImportProfile.profile;
              if (isOverTimeBudget) {
                return {
                  loading: false,
                  profile: dataImportProfile.profile,
                };
              }
            }
            dispatch(fetchProfile({ currentTaxReturn: args.currentTaxReturn }));
            return {
              loading: true,
            };
          }, 1000);
        }
        return {
          loading: shouldRetry,
          profile: processedResult,
        };
      } catch (error) {
        return rejectWithValue(`Caught issue with fetching profile`);
      }
    }

    if (dataImportMode === `clientside-intercept`) {
      return {
        loading: false,
        profile: getProfileForClientsideIntercept(),
      };
    }

    return rejectWithValue(`Disabled`);
  }
);

export type DataImportProfileState = {
  data:
    | {
        status: 'complete';
        profile: DataImportProfile;
        createdAt: string;
      }
    | {
        status: 'un-initialized' | 'fetching';
      }
    | {
        status: 'error';
        error: SerializedError;
      };
};

const initialState: DataImportProfileState = {
  data: {
    status: `un-initialized`,
  },
};

const dataImportProfileSlice = createSlice({
  name: `dataImportProfile`,
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchProfile.pending, (state) => {
        state.data = { status: `fetching` };
      })
      .addCase(fetchProfile.fulfilled, (state, action) => {
        const { profile, loading } = action.payload;
        if (loading) {
          state.data = { status: `fetching` };
        } else {
          state.data = {
            status: `complete`,
            profile,
            createdAt: new Date().toISOString(),
          };
        }
      })
      .addCase(fetchProfile.rejected, (state, action) => {
        state.data = {
          status: `error`,
          error: action.error,
        };
      });
  },
});

function getFetchStatusForProfile(profile: DataImportProfile): 'waiting-for-backend' | 'fetching' | 'settled' {
  const hasInFlightResponses = Object.values(profile.data).some((p) => p.state === `incomplete`);
  if (hasInFlightResponses) {
    return `waiting-for-backend`;
  }

  return `settled`;
}

export const selectDataImportFetchState = (state: {
  dataImportProfile: DataImportProfileState;
}): 'waiting-for-backend' | 'fetching' | 'settled' => {
  const profile = state.dataImportProfile.data;
  if (profile.status === `fetching` || profile.status === `un-initialized`) {
    return `fetching`;
  }
  if (profile.status === `error`) {
    return `settled`;
  }

  if (profile.status === `complete`) {
    if (profile.profile.isOverTimeBudget) {
      return `settled`;
    }
    const input = profile.profile;
    return getFetchStatusForProfile(input);
  }

  assertNever(profile.status);
};

export const dataImportProfileSliceReducer = dataImportProfileSlice.reducer;
