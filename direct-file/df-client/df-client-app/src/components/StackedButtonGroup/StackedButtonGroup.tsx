import styles from './StackedButtonGroup.module.scss';
import { ReactNode } from 'react';

type StackedButtonGroupProps = {
  children: ReactNode;
};
const StackedButtonGroup = ({ children }: StackedButtonGroupProps) => {
  return (
    <div className={styles.stackedButtonGroup} data-testid='stackedButtonGroup'>
      {children}
    </div>
  );
};

export default StackedButtonGroup;
