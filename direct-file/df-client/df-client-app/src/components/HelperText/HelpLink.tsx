import { FC } from 'react';
import { InfoDisplayProps } from '../../types/core.js';
import Translation from '../Translation/index.js';
import { useTranslation } from 'react-i18next';

const HelpLink: FC<InfoDisplayProps> = ({ i18nKey, collectionId }) => {
  const { i18n } = useTranslation();
  if (!i18n.exists(`info.${i18nKey}.helpText.helpLink`)) {
    return null;
  }
  return (
    <p>
      <Translation i18nKey={`info.${i18nKey}.helpText.helpLink`} collectionId={collectionId} />
    </p>
  );
};

export default HelpLink;
