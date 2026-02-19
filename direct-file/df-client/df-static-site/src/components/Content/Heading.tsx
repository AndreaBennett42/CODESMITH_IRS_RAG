import { ReactNode } from 'react';

type HeadingProps = {
  children: ReactNode | ReactNode[];
  splash?: boolean;
  level?: `h1` | `h2` | `h3` | `h4` | `h5` | `h6`;
  large?: boolean;
  customClassName?: string;
};

const Heading = ({ children, splash = false, level, large, customClassName }: HeadingProps) => {
  const H = level ? level : splash ? `h1` : `h2`;
  const typeSetScreenerHeading = splash || large;
  const props = {
    className: customClassName
      ? customClassName
      : typeSetScreenerHeading
      ? undefined
      : `font-body-lg text-normal margin-top-0`,
  };
  return (
    /* quick explanation of this div:
     * setting role to `log` changes this to an aria live region to help disabled
     * users with screen readers hear when a new "page" has loaded
     * there are several ways to accomplish this
     * this variant says "when new content is added (`additions`) to this div, read
     * the entire thing (`atomic=true`) but don't interrupt (`polite`)" */
    <div role='log' aria-live='polite' aria-relevant='additions' aria-atomic='true'>
      <H {...props}>{children}</H>
    </div>
  );
};

export default Heading;
