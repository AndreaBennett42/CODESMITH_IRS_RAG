import {
  COLLECTION_LABEL,
  FACT_LABELS_TRANSLATION_KEY_PREFIX,
  TRANSFERABLE_DATA_TRANSLATION_KEY_PREFIX,
} from './constants.js';

export const getCollectionLabelI18nKey = (collectionKey: string) =>
  // eslint-disable-next-line max-len
  `${TRANSFERABLE_DATA_TRANSLATION_KEY_PREFIX}.${FACT_LABELS_TRANSLATION_KEY_PREFIX}.${collectionKey}.${COLLECTION_LABEL}`;

export const getFactLabelI18nKey = (factKey: string, collectionKey?: string) =>
  collectionKey
    ? `${TRANSFERABLE_DATA_TRANSLATION_KEY_PREFIX}.${FACT_LABELS_TRANSLATION_KEY_PREFIX}.${collectionKey}.${factKey}`
    : `${TRANSFERABLE_DATA_TRANSLATION_KEY_PREFIX}.${FACT_LABELS_TRANSLATION_KEY_PREFIX}.${factKey}`;
