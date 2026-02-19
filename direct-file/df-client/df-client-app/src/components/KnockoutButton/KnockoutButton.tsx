import { Button } from '@trussworks/react-uswds';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { FC } from 'react';

type KnockoutButtonProps = {
  i18nKey: string;
};

const KnockoutButton: FC<KnockoutButtonProps> = ({ i18nKey }) => {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const buttonText = t(`${i18nKey}`);

  return (
    <div className='screen__actions'>
      <Button type='button' onClick={() => navigate(`/home`)}>
        {buttonText}
      </Button>
    </div>
  );
};

export default KnockoutButton;
