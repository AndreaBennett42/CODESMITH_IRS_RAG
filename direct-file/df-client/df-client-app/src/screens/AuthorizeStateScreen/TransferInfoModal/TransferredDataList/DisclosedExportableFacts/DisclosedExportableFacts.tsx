import { ExportableFacts } from '../../../../../utils/exportUtils.js';
import DisclosedExportableCollection from '../DisclosedExportableCollection/DisclosedExportableCollection.js';
import DisclosedExportableFact from '../DisclosedExportableFact/DisclosedExportableFact.js';

type DisclosedExportableProps = {
  exportableFacts: ExportableFacts;
};
const DisclosedExportableFacts = ({ exportableFacts }: DisclosedExportableProps) => {
  return (
    <dl>
      {Object.entries(exportableFacts).map(([key, value], index) => {
        return Array.isArray(value) ? (
          <DisclosedExportableCollection
            key={`disclosed-exportable-collection-${index}-${key}`}
            collectionKey={key}
            exportableCollection={value}
          />
        ) : (
          <DisclosedExportableFact key={`disclosed-exportable-fact-${index}-${key}`} factKey={key} fact={value} />
        );
      })}
    </dl>
  );
};

export default DisclosedExportableFacts;
