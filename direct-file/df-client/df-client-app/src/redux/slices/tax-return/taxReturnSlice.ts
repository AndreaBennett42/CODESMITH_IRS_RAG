import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { TaxReturnsContextType } from '../../../context/TaxReturnsContext.js';
import { TaxReturn } from '../../../types/core.js';
import { taxReturnFetch } from './taxReturnFetch.js';
import { SystemAlertKey } from '../../../context/SystemAlertContext/SystemAlertContext.js';
import { deleteSystemAlert, setSystemAlert } from '../system-alert/systemAlertSlice.js';
import { getCurrentTaxYearReturn } from '../../../utils/taxReturnUtils.js';
import type { WritableDraft } from 'immer';
import { AppDispatch, RootState } from '../../store.js';
import { taxReturnCreate } from './taxReturnCreate.js';
import { maybeGetTaxReturnFromUnknown } from '../../../context/TaxReturnSchema.js';
import { fetchProfile } from '../data-import/dataImportProfileSlice.js';
export const TAX_ID_STORAGE_KEY = `taxId`;

// TODO - this is a bit of a mess - maybe collapse down the context into here.
export type TaxReturnsSliceStateData = Pick<
  TaxReturnsContextType,
  'currentTaxReturnId' | 'taxReturns' | 'isFetching' | 'fetchSuccess'
> & {
  hasFetchError: boolean;
  creationFetch: {
    isFetching: boolean;
    fetchSuccess: boolean;
    hasFetchError: boolean;
  };
};

function isErrorWithStatus(error: unknown): error is { status: number } {
  return typeof error === `object` && error !== null && `status` in error && typeof error[`status`] === `number`;
}

type TaxReturnsSliceState = {
  data: TaxReturnsSliceStateData;
};

export const initialState: TaxReturnsSliceState = {
  data: {
    currentTaxReturnId: null,
    taxReturns: [],
    isFetching: false,
    fetchSuccess: false,
    hasFetchError: false,
    creationFetch: {
      isFetching: false,
      fetchSuccess: false,
      hasFetchError: false,
    },
  },
};

export const fetchTaxReturns = createAsyncThunk<
  // Return type of the payload creator
  Awaited<ReturnType<typeof taxReturnFetch>>,
  // First argument passed to the payload creator
  void,
  {
    dispatch: AppDispatch;
    state: RootState;
    rejectValue: `fetch-error`;
  }
>(`taxReturns/fetchTaxReturns`, async (_, { rejectWithValue, dispatch, getState }) => {
  try {
    // TODO - we should make system alerts for tax return fetch just be a function of the tax
    // return state rather than having to fire multiple actions here.
    //
    // https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/11963
    const result = await taxReturnFetch();

    const { taxReturns, currentTaxReturn } = extractTaxReturnValuesFromAPIResponse(result);
    const { currentTaxReturnId: existingCurrentTaxReturnID } = getState().taxReturns.data;
    const responseTaxReturnID = currentTaxReturn?.id;

    // TODO - actually handle the currentTaxReturn changing.
    if (currentTaxReturn && responseTaxReturnID !== existingCurrentTaxReturnID) {
      dispatch(fetchProfile({ currentTaxReturn }));
    }
    dispatch(deleteSystemAlert(SystemAlertKey.FETCH_TAX_RETURNS));

    // TODO - centralize handling of the updates the lead to changes in current tax return.
    return taxReturns;
  } catch (e) {
    // TODO - we should make system alerts for tax return fetch just be a function of the tax
    // return state rather than having to fire multiple actions here.
    dispatch(
      setSystemAlert({
        key: SystemAlertKey.FETCH_TAX_RETURNS,
        config: {
          type: `error`,
          i18nKey: `generic.serverError`,
        },
      })
    );
    return rejectWithValue(`fetch-error`);
  }
});

export const createTaxReturn = createAsyncThunk<
  // Return type of the payload creator
  Awaited<ReturnType<typeof taxReturnCreate>>,
  // First argument passed to the payload creator
  void,
  {
    dispatch: AppDispatch;
    state: RootState;
    rejectValue: `fetch-error` | `refreshing-returns`;
  }
>(`taxReturns/createTaxReturn`, async (_, { rejectWithValue, dispatch }) => {
  try {
    const creationResult = await taxReturnCreate();

    dispatch(fetchProfile({ currentTaxReturn: creationResult }));
    return creationResult;
  } catch (e) {
    if (isErrorWithStatus(e) && e.status === 409) {
      // 409 indicates return already exists; refresh the tax returns and it'll show up.
      dispatch(fetchTaxReturns());
      return rejectWithValue(`refreshing-returns`);
    } else {
      return rejectWithValue(`fetch-error`);
    }
  }
});

const taxReturnSlice = createSlice({
  name: `taxReturns`,
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchTaxReturns.pending, (state) => {
        state.data.isFetching = true;
        state.data.fetchSuccess = false;
        state.data.hasFetchError = false;
      })
      .addCase(
        fetchTaxReturns.fulfilled,
        (state: WritableDraft<TaxReturnsSliceState>, action: PayloadAction<TaxReturn[]>) => {
          state.data.isFetching = false;
          state.data.fetchSuccess = true;
          state.data.hasFetchError = false;
          writeTaxReturnsToWritableState(state, action.payload);
        }
      )
      .addCase(fetchTaxReturns.rejected, (state) => {
        state.data.isFetching = false;
        state.data.fetchSuccess = false;
        state.data.hasFetchError = true;
      })
      .addCase(createTaxReturn.pending, (state) => {
        state.data.creationFetch.isFetching = true;
        state.data.creationFetch.fetchSuccess = false;
        state.data.creationFetch.hasFetchError = false;
      })
      .addCase(
        createTaxReturn.fulfilled,
        (state: WritableDraft<TaxReturnsSliceState>, action: PayloadAction<TaxReturn>) => {
          state.data.creationFetch.isFetching = false;
          state.data.creationFetch.fetchSuccess = true;
          state.data.creationFetch.hasFetchError = false;
          // Send things through a list of entries to avoid duplicating a return.
          const newTaxReturns = Object.values(
            Object.fromEntries([
              ...state.data.taxReturns.map((tr): [string, TaxReturn] => [tr.id, tr]),
              [action.payload.id, action.payload],
            ])
          );
          writeTaxReturnsToWritableState(state, newTaxReturns);
        }
      )
      .addCase(createTaxReturn.rejected, (state, action) => {
        const rejectValue = action.payload;
        if (rejectValue === `fetch-error`) {
          state.data.creationFetch.isFetching = false;
          state.data.creationFetch.fetchSuccess = false;
          state.data.creationFetch.hasFetchError = true;
        }

        if (rejectValue === `refreshing-returns`) {
          state.data.isFetching = true;
          state.data.fetchSuccess = false;
          state.data.hasFetchError = false;
          state.data.creationFetch.isFetching = false;
          state.data.creationFetch.fetchSuccess = false;
          state.data.creationFetch.hasFetchError = false;
        }
      });
  },
});

function extractTaxReturnValuesFromAPIResponse(taxReturns: TaxReturn[]): {
  taxReturns: TaxReturn[];
  currentTaxReturn: TaxReturn | undefined;
} {
  taxReturns.forEach((tx) => {
    if (maybeGetTaxReturnFromUnknown(tx) === null) {
    }
  });

  const currentTaxReturn = getCurrentTaxYearReturn(taxReturns);
  return {
    taxReturns,
    currentTaxReturn,
  };
}

function writeTaxReturnsToWritableState(state: WritableDraft<TaxReturnsSliceState>, taxReturns: TaxReturn[]): void {
  state.data.taxReturns = taxReturns;
  if (taxReturns.length === 0) {
    state.data.currentTaxReturnId = null;
    return;
  }
  const currentYearTaxReturn = getCurrentTaxYearReturn(taxReturns);
  if (currentYearTaxReturn) {
    state.data.currentTaxReturnId = currentYearTaxReturn.id;
  } else {
    state.data.currentTaxReturnId = null;
  }
}

export const selectCurrentTaxReturn = (state: { taxReturns: TaxReturnsSliceState }): TaxReturn | undefined => {
  const currentTaxReturnID = state.taxReturns.data.currentTaxReturnId;
  return state.taxReturns.data.taxReturns.find((tr) => tr.id === currentTaxReturnID);
};

export const selectCurrentTaxReturnCreate = (state: {
  taxReturns: TaxReturnsSliceState;
}): { isFetching: boolean; fetchSuccess: boolean; hasFetchError: boolean } => {
  return state.taxReturns.data.creationFetch;
};

export const taxReturnSliceReducer = taxReturnSlice.reducer;
