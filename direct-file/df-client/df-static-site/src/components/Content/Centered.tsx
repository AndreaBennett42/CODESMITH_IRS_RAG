import { ReactNode } from 'react';

const Centered = ({ children }: { children: ReactNode | ReactNode[] }) => (
  <div className='margin-top-2 display-flex flex-column flex-align-center'>{children}</div>
);

export default Centered;
