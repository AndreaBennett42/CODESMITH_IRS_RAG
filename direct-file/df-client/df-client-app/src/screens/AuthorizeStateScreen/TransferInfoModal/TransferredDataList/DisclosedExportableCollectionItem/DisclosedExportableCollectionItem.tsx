import { ExportableCollectionItem } from '../../../../../utils/exportUtils.js';
import DisclosedExportableFact from '../DisclosedExportableFact/DisclosedExportableFact.js';

import styles from './DisclosedExportableCollectionItem.module.scss';

type DisclosedExportableCollectionItemProps = {
  collectionKey: string;
  exportableCollectionItem: ExportableCollectionItem;
} & JSX.IntrinsicElements['dl'];
const DisclosedExportableCollectionItem = ({
  collectionKey,
  exportableCollectionItem,
  ...dlProps
}: DisclosedExportableCollectionItemProps) => {
  return (
    <dl className={styles.descriptionList} {...dlProps}>
      {Object.entries(exportableCollectionItem).map(([factKey, exportableFact], index) => (
        <DisclosedExportableFact
          key={`disclosed-exportable-fact-${index}-${factKey}}`}
          factKey={factKey}
          fact={exportableFact}
          collectionKey={collectionKey}
        />
      ))}
    </dl>
  );
};

export default DisclosedExportableCollectionItem;
