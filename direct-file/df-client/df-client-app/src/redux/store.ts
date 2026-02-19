import { combineReducers, configureStore } from '@reduxjs/toolkit';
import { dataImportProfileSliceReducer } from './slices/data-import/dataImportProfileSlice.js';
import { systemAlertSliceReducer } from './slices/system-alert/systemAlertSlice.js';
import { telemetrySliceReducer } from './slices/telemetry/telemetrySlice.js';
import { taxReturnSliceReducer } from './slices/tax-return/taxReturnSlice.js';
import { electronicSignatureSliceReducer } from './slices/electronic-signature/electronicSignatureSlice.js';

const rootReducer = combineReducers({
  dataImportProfile: dataImportProfileSliceReducer,
  taxReturns: taxReturnSliceReducer,
  systemAlert: systemAlertSliceReducer,
  telemetry: telemetrySliceReducer,
  electronicSignature: electronicSignatureSliceReducer,
});

// Make it easy to reset and preload store values in unit tests
export const setupStore = (preloadedState?: Partial<RootState>) => {
  return configureStore({
    reducer: rootReducer,
    preloadedState,
  });
};

// Store value used throughout the application
export const store = setupStore();

export type RootState = ReturnType<typeof rootReducer>;
export type AppStore = ReturnType<typeof setupStore>;
export type AppDispatch = AppStore['dispatch'];
