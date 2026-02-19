import { exportToStateFacts } from '../../../../fact-dictionary/generated/exportToStateFacts.js';

export const COLLECTION_LABEL = `COLLECTION_LABEL`;

export const TRANSFERABLE_DATA_TRANSLATION_KEY_PREFIX = `authorizeState.transferableData`;
export const FACT_LABELS_TRANSLATION_KEY_PREFIX = `factLabels`;

export const SENSITIVE_EXPORTABLE_FACTS: (typeof exportToStateFacts)[number][] = [
  `/familyAndHousehold/*/tin`,
  `/filers/*/tin`,
  `/interestReports/*/recipientTin`,
  `/interestReports/*/payer/tin`,
  `/formW2s/*/usedTin`,
  `/socialSecurityReports/*/recipientTin`,
] as const;
