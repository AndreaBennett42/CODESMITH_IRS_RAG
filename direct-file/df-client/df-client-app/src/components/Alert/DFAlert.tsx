import { ComponentProps, useMemo, FC } from 'react';
import { Alert } from '@trussworks/react-uswds';
import Translation from '../Translation/index.js';
import InternalLink from '../InternalLink/index.js';
import { useTranslation } from 'react-i18next';
import { CommonTranslation } from 'df-i18n';
import './DFAlert.module.scss';
import { BareContentDisplay } from '../ContentDisplay/index.js';
import { i18n } from 'i18next';
import type { TranslationProps } from '../Translation/Translation.js';

type AlertProps = ComponentProps<typeof Alert>;

export type DFAlertProps = AlertProps & {
  i18nKey: string | null;
  collectionId: string | null;
  showTextAsHeader?: boolean;
  internalLink?: string;
  additionalComponents?: Record<string, JSX.Element>;
  context?: TranslationProps[`context`];
};

type KeyValuesResult = {
  headingValue?: JSX.Element;
  bodyValue?: JSX.Element;
};

const getKeyValues = (
  i18n: i18n,
  i18nKey: string,
  showTextAsHeader: boolean,
  isStandAloneKey: boolean,
  collectionId: string | null,
  additionalComponents?: Record<string, JSX.Element>,
  context?: TranslationProps[`context`]
): KeyValuesResult => {
  // alerts can--but aren't required to--have both a heading and a body
  let headingValue: JSX.Element | undefined;
  let bodyValue: JSX.Element | undefined;

  if (isStandAloneKey) {
    const valueAtKey = (
      <Translation i18nKey={i18nKey} collectionId={collectionId} components={additionalComponents} context={context} />
    );
    if (showTextAsHeader) {
      headingValue = valueAtKey;
    } else {
      bodyValue = valueAtKey;
    }
  } else {
    headingValue = i18n.exists(`${i18nKey}.alertText.heading`) ? (
      <Translation
        i18nKey={`${i18nKey}.alertText.heading`}
        collectionId={collectionId}
        components={additionalComponents}
        context={context}
      />
    ) : undefined;

    if (i18n.exists(`${i18nKey}.alertText.body`)) {
      bodyValue = (
        <BareContentDisplay
          collectionId={collectionId}
          i18nKey={`${i18nKey}.alertText`}
          additionalComponents={additionalComponents}
          context={context}
        />
      );
    }
  }

  return { headingValue, bodyValue };
};

const DFAlert: FC<DFAlertProps> = ({
  i18nKey,
  collectionId,
  type,
  showTextAsHeader,
  internalLink,
  context,
  additionalComponents,
  children,
  ...props
}) => {
  const translationComponents = useMemo(() => {
    if (i18nKey !== null && internalLink) {
      return {
        ...additionalComponents,
        InternalLink: <InternalLink i18nKey={i18nKey} collectionId={collectionId} route={internalLink} />,
      };
    } else {
      return additionalComponents;
    }
  }, [additionalComponents, collectionId, i18nKey, internalLink]);

  const { t, i18n } = useTranslation();
  const keyValues = useMemo((): ReturnType<typeof getKeyValues> => {
    if (i18nKey === null) return {};

    const namespacedKey = CommonTranslation.getNamespacedKey(i18nKey);
    if (!i18n.exists(namespacedKey)) {
      return {};
    }

    const isStandAloneKey = typeof t(namespacedKey, { returnObjects: true }) === `string`;
    return getKeyValues(
      i18n,
      namespacedKey,
      showTextAsHeader ?? false,
      isStandAloneKey,
      collectionId,
      translationComponents,
      context
    );
  }, [collectionId, context, i18n, i18nKey, showTextAsHeader, t, translationComponents]);

  const { headingValue, bodyValue } = keyValues;

  return (
    <Alert type={type} heading={headingValue} {...props} validation slim>
      {bodyValue}
      {children}
    </Alert>
  );
};

export default DFAlert;
