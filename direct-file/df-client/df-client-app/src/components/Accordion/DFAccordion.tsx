import { useTranslation } from 'react-i18next';
import { CommonAccordion, CommonAccordionProps } from '@irs/df-common';
import { CommonTranslation } from 'df-i18n';
import Translation from '../Translation/index.js';
import { useMemo } from 'react';
import { isBeforeResubmissionDeadline } from '../../utils/dateUtils.js';
import InternalLink from '../InternalLink/index.js';

const DFAccordion = ({
  i18nKey,
  internalLink,
  additionalComponents,
  collectionId,
  ...props
}: Omit<CommonAccordionProps, 'TranslationComponent'> & { internalLink?: string }) => {
  const { i18n } = useTranslation();
  const namespacedKey = CommonTranslation.getNamespacedKey(i18nKey);
  const innerAdditionalComponents = useMemo(() => {
    // Make sure the customersupportLink is available, if needed
    // CommonAccordion doesn't delegate to ContentDisplay
    const now = new Date();

    const innerAdditionalComponents: typeof additionalComponents = {
      ...additionalComponents,
    };

    if (internalLink) {
      innerAdditionalComponents[`InternalLink`] = (
        <InternalLink i18nKey={i18nKey} collectionId={collectionId ?? null} route={internalLink} />
      );
    }

    return innerAdditionalComponents;
  }, [additionalComponents, collectionId, i18nKey, internalLink]);

  if (!i18n.exists(namespacedKey)) {
    return null;
  }
  return (
    <CommonAccordion
      i18nKey={namespacedKey}
      collectionId={collectionId}
      additionalComponents={innerAdditionalComponents}
      TranslationComponent={Translation}
      {...props}
    />
  );
};

export default DFAccordion;
