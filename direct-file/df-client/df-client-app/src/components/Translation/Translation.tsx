import { FC, useMemo } from 'react';
import useTranslationContextFromFacts from '../../hooks/useTranslationContextFromFacts.js';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { Trans, useTranslation } from 'react-i18next';
import { ModalRef } from '@trussworks/react-uswds';
import { CommonTranslation } from 'df-i18n';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import { useFactGraphTranslationContext } from '../../context/FactGraphTranslationContext.js';
import { CURRENT_TAX_YEAR } from '../../constants/taxConstants.js';

export type I18nKey = string | string[];

export interface TranslationProps {
  i18nKey: I18nKey;
  collectionId: string | null;
  context?: object;
  components?: Record<string, JSX.Element>;
}

const NO_FG_CONTEXT_ITEMS: { [key: string]: string } = {
  '/taxYear': CURRENT_TAX_YEAR,
};

interface TranslationValues {
  [key: string]: string | number | Date;
}

export function buildLinkComponents(urls: Record<string, string>, modalRef?: React.RefObject<ModalRef>) {
  return Object.fromEntries(
    Object.entries(urls).map(([name, url]) => [name, <CommonLinkRenderer key={url} url={url} modalRef={modalRef} />])
  ) as Record<string, JSX.Element>;
}

const Translation: FC<TranslationProps> = ({ i18nKey, collectionId, context = {}, components }) => {
  const { t } = useTranslation();
  const { data, urls } = CommonTranslation.maybeUrls(t, i18nKey);
  const transKey = CommonTranslation.getTranslationKey(i18nKey, data);

  const { factGraph } = useFactGraph();
  const { shouldFetchTranslationValuesFromFactGraph } = useFactGraphTranslationContext();
  const getTranslationContext = useTranslationContextFromFacts(factGraph, collectionId);

  const values: TranslationValues = useMemo(() => {
    if (!shouldFetchTranslationValuesFromFactGraph) {
      return {
        ...NO_FG_CONTEXT_ITEMS,
        ...context,
      };
    }
    const translationContext = getTranslationContext(transKey);
    return { ...context, ...translationContext };
  }, [transKey, getTranslationContext, context, shouldFetchTranslationValuesFromFactGraph]);

  const linkComponents = buildLinkComponents(urls);

  const allowedComponents = {
    italic: <i />,
    strong: <strong />,
    ul: <ul />,
    li: <li />,
    ol: <ol />,
    br: <br />,
    ...linkComponents,
    ...components,
  };

  return <Trans i18nKey={transKey} values={values} components={allowedComponents} />;
};

export default Translation;
