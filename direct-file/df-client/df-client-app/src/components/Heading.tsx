import { InfoDisplayProps } from '../types/core.js';
import PageTitle from './PageTitle/index.js';
import { useTranslation } from 'react-i18next';
import { getModalOrTranslationComponent } from '../utils/modalHelpers.js';
import useTranslatePIIRedacted from '../hooks/useTranslatePIIRedacted.js';

const Heading = ({ i18nKey, collectionId }: InfoDisplayProps) => {
  const { t } = useTranslation();

  // Select the key to use for the title
  const nameSpace = i18nKey.startsWith(`/info`) ? `info.` : `headings.`;
  const namespacedKey = `${nameSpace}${i18nKey}`;
  const { Component, isModal } = getModalOrTranslationComponent(t, namespacedKey);
  const calculatedI18nKey = isModal ? `${i18nKey}` : namespacedKey;

  // Select the key to use for the redacted title
  const redactedI18nKey: string = isModal ? `${namespacedKey}.helpText.modals.text` : namespacedKey;

  const components = {
    // Headings with `W-2`, for example, may need a span in the translation in order to turn off default text wrapping.
    span: <span className='text-no-wrap' />,
  };

  return (
    <PageTitle redactedTitle={useTranslatePIIRedacted(redactedI18nKey, true)}>
      <Component components={components} i18nKey={calculatedI18nKey} collectionId={collectionId} />
    </PageTitle>
  );
};

export default Heading;
