import './components/CommonHeader/CommonHeader.module.scss';

export {
  CommonContentDisplay,
  buildLinkComponents,
  useLinkComponents,
} from './components/CommonContentDisplay/CommonContentDisplay.js';
export type { CommonContentDisplayProps } from './components/CommonContentDisplay/CommonContentDisplay.js';
export { default as CommonHeader } from './components/CommonHeader/index.js';
export { default as CommonFooter } from './components/CommonFooter/index.js';
export { CommonLanguageSelector } from './components/CommonLanguageSelector/CommonLanguageSelector.js';
export type { CommonLanguageSelectorProps } from './components/CommonLanguageSelector/CommonLanguageSelector.js';
export { default as CommonNotFound } from './components/CommonNotFound/index.js';
export { default as CommonAccordion } from './components/CommonAccordion/index.js';
export type { CommonAccordionProps } from './components/CommonAccordion/CommonAccordion.js';
export type { CommonAccordionItemProps } from './components/CommonAccordion/CommonAccordion.js';

export {
  CommonOverrideDatePicker,
  CommonOverrideBanner,
  CommonOverrideDobPicker,
  commonOverrideOverrideDateIfRequired,
} from './components/CommonOverrideTools/index.js';
