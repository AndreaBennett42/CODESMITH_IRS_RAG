import { InfoDisplayProps } from '../types/core.js';
import Translation from './Translation/index.js';

const Subheading = ({ i18nKey, collectionId }: InfoDisplayProps) => {
  return (
    <h2>
      <Translation i18nKey={i18nKey} collectionId={collectionId} />
    </h2>
  );
};

export default Subheading;
