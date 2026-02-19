import { DFAlertProps } from '../../components/Alert/DFAlert.js';
import { createContext, ReactNode, useCallback, useContext, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { CustomSystemAlertConfigBuilderOptions } from './customSystemAlertConfigs.js';
import {
  deleteSystemAlert,
  setSystemAlert,
  removeAlertsForLocationChange,
} from '../../redux/slices/system-alert/systemAlertSlice.js';
import { store } from '../../redux/store.js';
import { useAppSelector } from '../../redux/hooks.js';

export enum SystemAlertKey {
  USE_SAVE_AND_PERSIST = `USE_SAVE_AND_PERSIST`,
  USE_SAVE_AND_CONTINUE_AND_PERSIST = `USE_SAVE_AND_CONTINUE_AND_PERSIST`,
  SUBMIT = `SUBMIT`,
  RESET = `RESET`,
  FETCH_TAX_RETURNS = `FETCH_TAX_RETURNS`,
  FETCH_DATA_TO_IMPORT = `FETCH_DATA_TO_IMPORT`,
  PREVIEW = `PREVIEW`,
  CREATE_AUTHORIZATION_CODE = `CREATE_AUTHORIZATION_CODE`,
}

// TODO: We may want to store these by type in case type should influence the render order.
//       Today we only implement `error` type alerts, so not critical
export type SystemAlertConfigs = Partial<Record<SystemAlertKey, StoredSystemAlertConfig>>;

export type StoredSystemAlertConfig = {
  alertConfig: SystemAlertConfig;
  shouldClearOnRouteChange?: boolean;
  timestamp: number;
};

export type SetSystemAlertConfig = SystemAlertConfig & {
  shouldClearOnRouteChange?: boolean;
  customSystemAlertConfigBuilderOptions?: CustomSystemAlertConfigBuilderOptions;
};

export type SystemAlertContextType = {
  // We do not want folks manipulating this object directly, as state updates and therefore rerenders only happen by
  // using the provided util callbacks
  systemAlerts: SystemAlertConfigs;
  setSystemAlert: (key: SystemAlertKey, config: SetSystemAlertConfig) => void;
  deleteSystemAlert: (key: SystemAlertKey) => void;
};

export const SYSTEM_ALERT_I18N_PREFIX = `systemAlerts`;

export type SystemAlertConfig = Omit<
  /**
   * i18nKey will be prefixed with I18N_KEY_PREFIX
   */
  DFAlertProps,
  `headingLevel` | `collectionId`
> & { i18nKey: string };

export const SystemAlertContext = createContext<SystemAlertContextType>({} as SystemAlertContextType);

type SystemAlertContextProviderProps = {
  children: ReactNode;
};

export function getEmptySystemAlertsMap(): Partial<Record<SystemAlertKey, StoredSystemAlertConfig>> {
  return {};
}

export const SystemAlertContextProvider = ({ children }: SystemAlertContextProviderProps) => {
  const systemAlerts = useAppSelector((state) => state.systemAlert.data);
  const location = useLocation();
  const setSystemAlertCallback = useCallback(
    (key: SystemAlertKey, config: SetSystemAlertConfig) => store.dispatch(setSystemAlert({ key, config })),
    []
  );

  // Clear system alerts on path change
  useEffect(() => {
    store.dispatch(removeAlertsForLocationChange());
  }, [location.pathname]);

  // TODO: Return a boolean value for whether or not a system alert has been deleted,
  //  but do not add any deps to the callback
  const deleteSystemAlertCallback = useCallback((alertKey: SystemAlertKey) => {
    store.dispatch(deleteSystemAlert(alertKey));
  }, []);

  const value: SystemAlertContextType = {
    systemAlerts,
    setSystemAlert: setSystemAlertCallback,
    deleteSystemAlert: deleteSystemAlertCallback,
  };

  return <SystemAlertContext.Provider value={value}>{children}</SystemAlertContext.Provider>;
};

export const useSystemAlertContext = () => useContext(SystemAlertContext);
