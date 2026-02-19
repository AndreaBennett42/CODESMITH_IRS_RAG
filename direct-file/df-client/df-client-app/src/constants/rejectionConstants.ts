export const MEF_REJECTION_ERROR_CODES = {
  /* 
    If a federal tax return gets rejected by any of the MeF errors below, 
    the taxpayer will be unable to edit and resubmit in Direct File because 
    the Forms required to resolve these errors are not supported by Direct File 
    in this current phase.
  */
  UNFIXABLE_BY_DF: [`S2-F1040-147`, `S2-F1040-429`],
};
