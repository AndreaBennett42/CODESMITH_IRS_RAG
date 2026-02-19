import { FC, useMemo } from 'react';
import TaxReturnAlert, { TaxReturnAlertProps } from './TaxReturnAlert.js';
import { useTranslation } from 'react-i18next';

const I18N_ROOT = `mefAlerts`;
type RenderLocation = 'data-view';

export type ContentDeclarationMefAlertProps = Omit<TaxReturnAlertProps, 'heading' | 'collectionId' | 'headingLevel'> & {
  type: `error` | `warning`;
  mefErrorCode: string;
  i18nKey: NonNullable<TaxReturnAlertProps[`i18nKey`]>;
  headingI18nKey?: string;
};

// There may be props that we don't wish to expose to flow config files, but are useful for rendering in specific
// scenarios. This type has additional props that may be manually specified when rendering an MefAlert
export type MefAlertProps = ContentDeclarationMefAlertProps & {
  renderLocation?: RenderLocation;
  collectionId: TaxReturnAlertProps['collectionId'];
  headingLevel?: TaxReturnAlertProps['headingLevel'];
};

const MefAlert: FC<MefAlertProps> = ({
  mefErrorCode,
  i18nKey,
  headingI18nKey,
  renderLocation,
  ...passThroughProps
}) => {
  const { t, i18n } = useTranslation(`translation`);
  const { type } = passThroughProps;

  const heading = useMemo(() => {
    const showHeading = !renderLocation;
    if (showHeading) {
      if (headingI18nKey && i18n.exists(headingI18nKey)) {
        return t(headingI18nKey);
      }

      const genericHeadingI18nKey = `${I18N_ROOT}.generic.heading.${type}`;
      if (i18n.exists(genericHeadingI18nKey)) {
        return t(genericHeadingI18nKey);
      }
    }

    return null;
  }, [renderLocation, headingI18nKey, i18n, t, type]);

  const translationKey = useMemo(() => {
    if (renderLocation) {
      const defaultRenderLocationI18nKey = `${I18N_ROOT}.generic.${renderLocation}.${type}`;
      const customRenderLocationI18nKey = `${I18N_ROOT}.${mefErrorCode}.${i18nKey}.${renderLocation}`;
      return i18n.exists(customRenderLocationI18nKey) ? customRenderLocationI18nKey : defaultRenderLocationI18nKey;
    }

    return `${I18N_ROOT}.${mefErrorCode}.${i18nKey}`;
  }, [renderLocation, mefErrorCode, i18nKey, type, i18n]);

  return <TaxReturnAlert i18nKey={translationKey} heading={heading} {...passThroughProps} />;
};

export default MefAlert;
