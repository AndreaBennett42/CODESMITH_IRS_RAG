// These purely exist to make mocking import.meta.env in tests easier without adding any 3rd party deps
export const isDev = () => import.meta.env.DEV;
export const getViteSadiAuthId = () => import.meta.env.VITE_SADI_AUTH_ID;
export const getViteSadiXffHeader = () => import.meta.env.VITE_SADI_XFF_HEADER;
export const getViteSadiTidHeader = () => import.meta.env.VITE_SADI_TID_HEADER;
