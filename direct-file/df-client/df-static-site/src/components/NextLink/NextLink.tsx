import { Link } from 'react-router-dom';

const NextLink = ({ children, href }: { children: string; href: string }) => (
  <div className='margin-y-2'>
    <Link className='usa-button' to={href}>
      {children}
    </Link>
  </div>
);

export default NextLink;
