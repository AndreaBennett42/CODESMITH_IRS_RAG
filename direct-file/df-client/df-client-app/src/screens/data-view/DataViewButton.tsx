import { Button, Icon } from '@trussworks/react-uswds';
import Translation from '../../components/Translation/index.js';
import { useNavigate } from 'react-router-dom';

type DataViewButtonProps = {
  route: string;
  isOutline: boolean;
  i18nKey?: string;
};

export const DataViewButton = ({
  route,
  isOutline = false,
  i18nKey = `button.go-to-checklist`,
}: DataViewButtonProps) => {
  const navigate = useNavigate();
  const IconComponent = isOutline ? Icon.NavigateFarNext : Icon.NavigateNext;

  return (
    <Button type={`button`} outline={isOutline} onClick={() => navigate(route !== `` ? route : `/checklist`)}>
      <Translation i18nKey={i18nKey} collectionId={null} />
      <IconComponent size={3} className='usa-button__icon-right' aria-hidden='true' />
    </Button>
  );
};
