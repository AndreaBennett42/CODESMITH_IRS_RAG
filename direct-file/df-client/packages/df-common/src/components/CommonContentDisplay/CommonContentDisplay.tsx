import { FunctionComponent, useMemo } from 'react';
import { TranslationProps, generateContent } from './contentGenerator.js';
import { useTranslation } from 'react-i18next';
import { CommonTranslation } from 'df-i18n';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import { ModalRef } from '@trussworks/react-uswds';

export type CommonContentDisplayProps = {
  i18nKey: string;
  collectionId?: string | null;
  allowedTags?: string[] | undefined;
  additionalComponents?: Record<string, JSX.Element>;
  TranslationComponent: FunctionComponent<TranslationProps>;
  /** subKey is used as `${i18nKey}.${subKey}`; the default is `body` */
  subKey?: string;
  /** noBody means don't use `subKey` */
  noBody?: boolean;
  context?: object;
};

export function buildLinkComponents(urls: Record<string, string>, modalRef?: React.RefObject<ModalRef>) {
  return Object.fromEntries(
    Object.entries(urls).map(([name, url]) => [name, <CommonLinkRenderer key={url} url={url} modalRef={modalRef} />])
  ) as Record<string, JSX.Element>;
}

export function useLinkComponents(baseKey: string) {
  const { t } = useTranslation();
  return useMemo(() => {
    const { urls } = CommonTranslation.maybeUrls(t, baseKey);
    return buildLinkComponents(urls);
  }, [t, baseKey]);
}

export const CommonContentDisplay = ({
  i18nKey,
  collectionId,
  allowedTags,
  additionalComponents,
  TranslationComponent,
  noBody = false,
  subKey = `body`,
  context = {},
}: CommonContentDisplayProps) => {
  const { t } = useTranslation();
  const baseKey = i18nKey.startsWith(`/info/`) ? `info.${i18nKey}` : i18nKey;

  // Check for any provided urls so they can be made available for use in any subobjects
  const linkComponents = useLinkComponents(baseKey);

  const generatedContent = useMemo(() => {
    const tKey = noBody ? baseKey : `${baseKey}.${subKey}`;
    const body = t(tKey, { returnObjects: true });
    return generateContent(
      tKey,
      body,
      collectionId || null,
      allowedTags,
      TranslationComponent,
      {
        ...linkComponents,
        ...additionalComponents,
      },
      context
    );
  }, [
    noBody,
    baseKey,
    subKey,
    t,
    collectionId,
    allowedTags,
    TranslationComponent,
    linkComponents,
    additionalComponents,
    context,
  ]);

  return generatedContent;
};
