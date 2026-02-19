import classNames from 'classnames';
import { ReactNode } from 'react';

interface ProseProps {
  children: ReactNode | ReactNode[];
  className?: string;
}

const Prose = ({ children, className = `` }: ProseProps) => {
  const classnames = classNames(`usa-prose`, className);

  return <div className={classnames}>{children}</div>;
};

export default Prose;
