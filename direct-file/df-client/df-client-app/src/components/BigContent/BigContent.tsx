import { InfoDisplayProps } from '../../types/core.js';
import ContentDisplay from '../ContentDisplay/index.js';
import styles from './BigContent.module.scss';

const BigContent = (props: InfoDisplayProps) => {
  const allowedTags: string[] = [`p`];

  return (
    <div className={styles.dfBigContent}>
      <ContentDisplay {...props} allowedTags={allowedTags} />
    </div>
  );
};

export default BigContent;
