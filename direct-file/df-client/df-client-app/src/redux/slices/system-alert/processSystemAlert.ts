import { CUSTOM_SYSTEM_ALERT_CONFIGS } from '../../../context/SystemAlertContext/customSystemAlertConfigs.js';
import {
  SetSystemAlertConfig,
  StoredSystemAlertConfig,
  SYSTEM_ALERT_I18N_PREFIX,
  SystemAlertKey,
} from '../../../context/SystemAlertContext/SystemAlertContext.js';

const DEFAULT_SHOULD_CLEAR_ON_ROUTE_CHANGE = true;

export function processSystemAlert(alertKey: SystemAlertKey, config: SetSystemAlertConfig): StoredSystemAlertConfig {
  // Get the configuration params passed from calling component
  const {
    shouldClearOnRouteChange = DEFAULT_SHOULD_CLEAR_ON_ROUTE_CHANGE,
    customSystemAlertConfigBuilderOptions,
    i18nKey,
    context,
    additionalComponents,
    internalLink,
    ...passThroughSystemAlertConfig
  } = config;

  // Check if provided i18nKey actually has a customization registered instead
  const customConfigValue = CUSTOM_SYSTEM_ALERT_CONFIGS.get(i18nKey) || {};

  // Custom alerts can return a config or a function that generates a config
  const customConfig =
    typeof customConfigValue === `function`
      ? customConfigValue(customSystemAlertConfigBuilderOptions)
      : customConfigValue;

  // Get the configuration params from the custom configuration
  const {
    nestedI18nKey,
    context: customContext,
    additionalComponents: customAdditionalComponents,
    internalLink: customInternalLink,
  } = customConfig;

  // If a nestedKey was provided use that
  const systemAlertI18nKey = nestedI18nKey
    ? `${SYSTEM_ALERT_I18N_PREFIX}.${i18nKey}.${nestedI18nKey}`
    : `${SYSTEM_ALERT_I18N_PREFIX}.${i18nKey}`;

  // Merge the passed in and custom parameters intelligently, notably with
  // custom parameters overriding defaults when provided
  const systemAlertContext = (context || customContext) && {
    ...context,
    ...customContext,
  };

  const systemAlertAdditionalComponents = (additionalComponents || customAdditionalComponents) && {
    ...additionalComponents,
    ...customAdditionalComponents,
  };

  const systemAlertInternalLink = customInternalLink ? customInternalLink : internalLink;

  // Create a new alert config from the results
  const systemAlertConfig: StoredSystemAlertConfig = {
    alertConfig: {
      i18nKey: systemAlertI18nKey,
      context: systemAlertContext,
      additionalComponents: systemAlertAdditionalComponents,
      internalLink: systemAlertInternalLink,
      ...passThroughSystemAlertConfig,
    },
    shouldClearOnRouteChange,
    timestamp: Date.now(),
  };
  return systemAlertConfig;
}
