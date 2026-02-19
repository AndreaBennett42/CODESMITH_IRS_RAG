import type { ErrorMessage } from '@trussworks/react-uswds';
import classnames from 'classnames';

/**
 * Stylistically identical to trusswork's ErrorMessage component, but without the alert role
 */
export const QuietErrorMessage: typeof ErrorMessage = ({ children, className, id }) => {
  const classes = classnames(`usa-error-message`, className);

  return (
    <span data-testid='errorMessage' className={classes} id={id}>
      {children}
    </span>
  );
};
