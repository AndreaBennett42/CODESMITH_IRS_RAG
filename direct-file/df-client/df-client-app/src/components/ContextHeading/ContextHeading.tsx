import { InfoDisplayProps } from '../../types/core.js';
import Translation from '../Translation/index.js';
import styles from './ContextHeading.module.scss';

export function ContextHeading({ i18nKey, collectionId }: InfoDisplayProps) {
  return (
    <div className={styles.contextHeading}>
      <Translation i18nKey={`headings.${i18nKey}`} collectionId={collectionId} />
    </div>
  );
}
