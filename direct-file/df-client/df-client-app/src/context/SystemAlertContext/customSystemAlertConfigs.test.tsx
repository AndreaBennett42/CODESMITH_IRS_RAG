import { I18nAlertKeysWithCustomConfig } from './customSystemAlertConfigs.js';
import { initI18n } from '../../i18n.js';
import { SYSTEM_ALERT_I18N_PREFIX } from './SystemAlertContext.js';
import { i18n as i18nType } from 'i18next';
import { expect } from 'vitest';

describe(`I18nAlertKeysWithCustomConfig`, () => {
  let i18n: i18nType;
  beforeAll(async () => {
    i18n = await initI18n();
  });

  it.each(Object.values(I18nAlertKeysWithCustomConfig))(`%s entry exists in the en.yaml file`, (enumValue) => {
    const translationKey = `${SYSTEM_ALERT_I18N_PREFIX}.${enumValue}`;

    const exists = i18n.exists(translationKey);

    expect(exists, `Expected ${translationKey} to exist, but it was not found`).toBeTruthy();
  });
});
