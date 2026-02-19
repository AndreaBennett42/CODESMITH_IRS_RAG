import { useCallback, useContext } from 'react';
import { useFactGraph } from '../factgraph/FactGraphContext.js';
import { save } from '../hooks/useApiHook.js';
import {
  SystemAlertKey,
  SetSystemAlertConfig,
  useSystemAlertContext,
} from '../context/SystemAlertContext/SystemAlertContext.js';
import { TaxReturnsContext } from '../context/TaxReturnsContext.js';

export const useSaveAndPersist = () => {
  const { setSystemAlert, deleteSystemAlert } = useSystemAlertContext();
  const { currentTaxReturnId } = useContext(TaxReturnsContext);

  const { factGraph } = useFactGraph();

  return useCallback((): Promise<{ hasPersistError: boolean }> => {
    const url = `${import.meta.env.VITE_BACKEND_URL}v1/taxreturns/${currentTaxReturnId}`;
    // eslint-disable-next-line df-rules/no-factgraph-save
    factGraph.save();
    const alertKey: SystemAlertKey = SystemAlertKey.USE_SAVE_AND_PERSIST;
    return save(url, { body: { facts: JSON.parse(factGraph.toJSON()) } })
      .then(() => {
        deleteSystemAlert(alertKey);
        return {
          hasPersistError: false,
        };
      })
      .catch((e) => {
        const config: SetSystemAlertConfig = {
          type: `error`,
          i18nKey: `generic.serverError`,
        };
        setSystemAlert(alertKey, config);
        return {
          hasPersistError: true,
        };
      });
  }, [deleteSystemAlert, factGraph, setSystemAlert, currentTaxReturnId]);
};
