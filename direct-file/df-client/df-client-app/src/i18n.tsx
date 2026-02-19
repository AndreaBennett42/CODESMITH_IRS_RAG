import { commonInitI18n } from 'df-i18n';

import { resources } from './locales/index.js';

export function initI18n() {
  return commonInitI18n(resources);
}
