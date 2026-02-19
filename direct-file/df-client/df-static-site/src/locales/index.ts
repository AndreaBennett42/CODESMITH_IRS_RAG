import englishTranslation from './en.yaml';
import spanishTranslation from './es.yaml';
// import pseudoTranslation from './px.json';

export { YamlSettings } from './yaml-settings.js';

export const resources = {
  en: {
    translation: englishTranslation,
  },
  es: {
    translation: spanishTranslation,
  },
} as const;
