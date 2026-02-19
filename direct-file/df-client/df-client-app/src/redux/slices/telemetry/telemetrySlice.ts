import { createSlice, createAsyncThunk, PayloadAction, ThunkAction, UnknownAction } from '@reduxjs/toolkit';
import type { AnyCodedExceptionConfig, CodedException, CodedExceptionConfig } from './codedException.js';
import { v4 as uuidv4 } from 'uuid';
import type { RootState } from '../../store.js';
import { isTelemetryEnabled } from '../../../constants/pageConstants.js';
import { StoredLoggedError } from './types.js';
import { processError } from './processError.js';

const SIX_HOURS_MS = 21600000;

type TelemetryState = {
  data: {
    inflightErrors: StoredLoggedError[];
    sentErrors: string[];
    flushState: 'idle' | 'pending';
  };
};

export const initialState: TelemetryState = {
  data: {
    inflightErrors: [],
    sentErrors: [],
    flushState: `idle`,
  },
};

function createError(config: AnyCodedExceptionConfig): StoredLoggedError {
  const id = uuidv4();
  const timestampSeconds = Math.floor(Date.now() / 1000).toString();
  const { key, payload } = config;
  return {
    id,
    timestampSeconds,
    key,
    error: processError(config.error),
    payload,
  };
}

const flushErrors = createAsyncThunk(
  `telemetry/flushErrors`,
  async (nextRetryDelay: number, { getState, rejectWithValue, dispatch }) => {
    const state = getState() as RootState;
    const errors = state.telemetry.data.inflightErrors;
    const hasErrorsToFlush = errors.length > 0;
    if (!hasErrorsToFlush) {
      return [];
    }

    try {
      // Fire another action in case more errors have come in.
      dispatch(flushErrors(nextRetryDelay));
      return errors.map((e) => e.id);
    } catch (error) {
      const truncatedNextRetryDelay = Math.min(nextRetryDelay, SIX_HOURS_MS);

      setTimeout(() => {
        dispatch(flushErrors(truncatedNextRetryDelay * 2));
      }, truncatedNextRetryDelay);

      return rejectWithValue(`Failed to send errors`);
    }
  },
  {
    condition: (arg, { getState }) => {
      const state = getState() as RootState;
      if (state.telemetry.data.flushState === `pending`) {
        // We're already sending telemetry in this case. We shouldn't start another request.
        // The telemetry thunk will enqueue any aditional failures on a second pass.
        return false;
      }

      return true;
    },
  }
);

const telemetrySlice = createSlice({
  name: `telemetrySlice`,
  initialState,
  reducers: {
    addCodedErrorToQueue: (state, action: PayloadAction<AnyCodedExceptionConfig>) => {
      const storedError = createError(action.payload);
      state.data.inflightErrors = [...state.data.inflightErrors, storedError];
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(flushErrors.fulfilled, (state, action) => {
        const newSentErrors = action.payload;
        const sentErrorSet = new Set([...state.data.sentErrors, ...newSentErrors]);
        state.data.sentErrors = [...sentErrorSet];
        state.data.inflightErrors = state.data.inflightErrors.filter((e) => !newSentErrors.includes(e.id));
        state.data.flushState = `idle`;
      })
      .addCase(flushErrors.rejected, () => {
        // Nothing to do - retry triggered in flushErrors.
      })
      .addCase(flushErrors.pending, (state) => {
        state.data.flushState = `pending`;
      });
  },
});

export const telemetrySliceReducer = telemetrySlice.reducer;
