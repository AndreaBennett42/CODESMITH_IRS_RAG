import { TFunction } from 'i18next';
import DFModal from '../components/HelperText/DFModal.js';
import Translation from '../components/Translation/index.js';
import InternalLink from '../components/InternalLink/index.js';
import { I18nKey } from '../components/Translation/Translation.js';
import { checkIfHasInternalLink, checkIfModal } from './i18nUtils.js';

// This function looks in the translation to figure out what kind of component should be used
// to render this key.
// Component: If it's a modal then Component will return a Modal
//            If it's not a modal Component will return a Translation
// hasInternalLink: True if i18nKey points to an object containing a key 'internalLink'
// isModal: True if hasInternalLink is false AND i18nKey points to an object containing 'helpText'
// InternalLink: Always returns an InternalLink
export const getModalOrTranslationComponent = (t: TFunction, key: I18nKey) => {
  // Get the translation value from i18n (could be an object)
  const data = t(key, { returnObjects: true });
  const isModal = checkIfModal(data);
  const hasInternalLink = checkIfHasInternalLink(data);
  const Component = checkIfModal(data) ? DFModal : Translation;

  return { Component, isModal, hasInternalLink, InternalLink };
};
