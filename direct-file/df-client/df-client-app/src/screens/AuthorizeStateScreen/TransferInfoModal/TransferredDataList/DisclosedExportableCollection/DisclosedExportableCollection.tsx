import { ExportableCollection } from '../../../../../utils/exportUtils.js';
// eslint-disable-next-line max-len
import DisclosedExportableCollectionItem from '../DisclosedExportableCollectionItem/DisclosedExportableCollectionItem.js';

import styles from './DisclosedExportableCollection.module.scss';
import { useTranslation } from 'react-i18next';

import { useMemo } from 'react';
import Translation from '../../../../../components/Translation/Translation.js';
import { getCollectionLabelI18nKey } from '../utils.js';

type DisclosedExportableCollectionProps = {
  collectionKey: string;
  exportableCollection: ExportableCollection;
};
const DisclosedExportableCollection = ({ collectionKey, exportableCollection }: DisclosedExportableCollectionProps) => {
  const { t, i18n } = useTranslation(`translation`);

  const id = `${collectionKey}-term`;

  const collectionI18nKey = useMemo(() => getCollectionLabelI18nKey(collectionKey), [collectionKey]);
  const collectionI18nKeyExists = i18n.exists(collectionI18nKey);
  const collectionLabel = collectionI18nKeyExists ? (
    <Translation i18nKey={collectionI18nKey} collectionId={null} />
  ) : (
    collectionKey
  );

  return (
    <>
      <dt id={id} className={styles.descriptionTerm}>
        {collectionLabel}
      </dt>
      <dd aria-labelledby={id} className={styles.descriptionDefinition}>
        {exportableCollection.map((exportableCollectionItem, index) => (
          <DisclosedExportableCollectionItem
            aria-label={`${t(collectionI18nKey)} ${index + 1}:`}
            key={`disclosed-exportable-collection-item-${collectionKey}-${index}`}
            collectionKey={collectionKey}
            exportableCollectionItem={exportableCollectionItem}
          />
        ))}
      </dd>
    </>
  );
};

export default DisclosedExportableCollection;
