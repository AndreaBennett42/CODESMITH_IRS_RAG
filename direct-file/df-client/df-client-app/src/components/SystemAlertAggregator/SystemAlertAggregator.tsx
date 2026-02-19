import DFAlert from '../Alert/DFAlert.js';
import screenStyles from '../Screen.module.scss';
import { useSystemAlertContext } from '../../context/SystemAlertContext/SystemAlertContext.js';
import { useMemo } from 'react';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { useContainsFactGraphValues } from '../../hooks/useContainsFactGraphValues.js';

const SystemAlertAggregator = () => {
  const { systemAlerts } = useSystemAlertContext();
  const { factGraph } = useFactGraph();
  const containsFactGraphValues = useContainsFactGraphValues();

  const systemAlertConfigs = useMemo(() => {
    const alertConfigs = Object.entries(systemAlerts)
      .map(([_key, config]) => config)
      .sort((a, b) => a.timestamp - b.timestamp)
      .map((storedConfig) => storedConfig.alertConfig);

    if (factGraph === undefined) {
      // If we don't have the fact graph context, skip rendering alerts that depend on it for their translations:
      return alertConfigs.filter((systemAlertConfig) => !containsFactGraphValues(systemAlertConfig.i18nKey));
    } else {
      return alertConfigs;
    }
  }, [containsFactGraphValues, factGraph, systemAlerts]);

  return systemAlertConfigs.map((systemAlertProps, index) => (
    <DFAlert
      key={`${systemAlertProps.i18nKey}-${index}`}
      headingLevel='h2'
      slim
      className={screenStyles.usaAlert}
      collectionId={null}
      data-testid={`system-alert`}
      {...systemAlertProps}
    />
  ));
};

export default SystemAlertAggregator;
