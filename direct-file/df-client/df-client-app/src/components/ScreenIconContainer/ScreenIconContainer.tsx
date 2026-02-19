import { ReactNode } from 'react';
import classNames from 'classnames';

import styles from './ScreenIconContainer.module.scss';

type IconContainerProps = {
  children: ReactNode;
} & JSX.IntrinsicElements['div'];

/**
 *
 * Note: defaults to aria-hidden="true"
 */
const ScreenIconContainer = ({ children, className, 'aria-hidden': ariaHidden, ...divProps }: IconContainerProps) => {
  const classes = classNames(styles.iconContainer, className);
  ariaHidden = ariaHidden === undefined ? true : ariaHidden;

  return (
    <div className={classes} aria-hidden={ariaHidden} {...divProps}>
      {children}
    </div>
  );
};

export default ScreenIconContainer;
