import { IconList } from '@trussworks/react-uswds';

interface DFIconListProps {
  children: React.ReactNode | React.ReactNode[];
}

const DFIconList = ({ children }: DFIconListProps) => (
  <div className='margin-top-3'>
    <IconList>{children}</IconList>
  </div>
);

export default DFIconList;
