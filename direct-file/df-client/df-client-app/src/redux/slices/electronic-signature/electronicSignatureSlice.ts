import { createSlice } from '@reduxjs/toolkit';

type ElectronicSignatureState = {
  electronicSigningFailed: boolean;
};

export const initialState: ElectronicSignatureState = {
  electronicSigningFailed: false,
};

export const electronicSignatureSlice = createSlice({
  name: `electronicSignature`,
  initialState,
  reducers: {
    setElectronicSignatureFailure: (state) => {
      state.electronicSigningFailed = true;
    },
    resetElectronicSignatureFailure: (state) => {
      state.electronicSigningFailed = false;
    },
  },
});

export const { setElectronicSignatureFailure, resetElectronicSignatureFailure } = electronicSignatureSlice.actions;
export const electronicSignatureSliceReducer = electronicSignatureSlice.reducer;
