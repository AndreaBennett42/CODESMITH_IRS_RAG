import { createSlice, PayloadAction, ThunkAction, UnknownAction } from '@reduxjs/toolkit';
import { createSelector } from 'reselect';
import { processSystemAlert } from './processSystemAlert.js';
import {
  SetSystemAlertConfig,
  StoredSystemAlertConfig,
  SystemAlertKey,
} from '../../../context/SystemAlertContext/SystemAlertContext.js';
import { RootState } from '../../store.js';

type SystemAlertState = {
  data: Partial<Record<SystemAlertKey, StoredSystemAlertConfig>>;
};
export const initialSystemAlertState: SystemAlertState = {
  data: {},
};

const systemAlertSlice = createSlice({
  name: `systemAlert`,
  initialState: initialSystemAlertState,
  reducers: {
    addSystemAlertToStorage: (state, action: PayloadAction<{ key: SystemAlertKey; config: SetSystemAlertConfig }>) => {
      const { key, config } = action.payload;
      const storedSystemAlert = processSystemAlert(key, config);

      state.data = {
        ...state.data,
        [key]: storedSystemAlert, // Safely add or update the alert
      };
    },
    deleteSystemAlert: (state, action: PayloadAction<SystemAlertKey>) => {
      const key = action.payload;
      const existingConfig = state.data[key];
      if (!existingConfig) {
      } else {
        delete state.data[key]; // Remove the alert
      }
    },
    removeAlertsForLocationChange: (state) => {
      state.data = Object.fromEntries(
        Object.entries(state.data).filter(([_key, systemAlert]) => !systemAlert.shouldClearOnRouteChange)
      );
    },
  },
});

// Input selector: extracts systemAlert.data from state
const selectSystemAlertData = (state: { systemAlert: SystemAlertState }) => state.systemAlert.data;

// Memoized selector
export const selectSortedSystemAlerts = createSelector([selectSystemAlertData], (data) => {
  return Object.entries(data)
    .map(([_key, config]) => config)
    .sort((a, b) => a.timestamp - b.timestamp); // Sort by timestamp in ascending order
});

export function setSystemAlert(systemAlertConfig: {
  key: SystemAlertKey;
  config: SetSystemAlertConfig;
}): ThunkAction<void, RootState, unknown, UnknownAction> {
  return (dispatch) => {
    dispatch(systemAlertSlice.actions.addSystemAlertToStorage(systemAlertConfig));
  };
}

export const { deleteSystemAlert, removeAlertsForLocationChange } = systemAlertSlice.actions;
export const systemAlertSliceReducer = systemAlertSlice.reducer;
