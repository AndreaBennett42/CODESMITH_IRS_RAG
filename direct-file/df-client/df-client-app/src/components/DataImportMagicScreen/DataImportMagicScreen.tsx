import { FC, ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { Button, Icon } from '@trussworks/react-uswds';
import { useNavigate } from 'react-router-dom';
import PageTitle from '../PageTitle/PageTitle.js';
import useTranslatePIIRedacted from '../../hooks/useTranslatePIIRedacted.js';
import { CURRENT_TAX_YEAR } from '../../constants/taxConstants.js';
import { BackButton } from '../../screens/ScreenHeader.js';

const translationContext: { [key: string]: string } = {
  '/taxYear': CURRENT_TAX_YEAR,
};

/**
 * Screen to display between Home and Checklist
 *
 */
export const DataImportMagicScreen: FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const redacted = useTranslatePIIRedacted(`dataImportMagicScreen.header`, true);

  return (
    <>
      <BackButton />
      <div className='display-flex flex-justify-center'>
        <img src={`${import.meta.env.VITE_PUBLIC_PATH}/imgs/sparkle.svg`} className='height-8 width-8' alt={``} />
      </div>
      <span className='screen__header'>
        <PageTitle redactedTitle={redacted} large>
          {t(`dataImportMagicScreen.header`, translationContext)}
        </PageTitle>
      </span>
      <p>{t(`dataImportMagicScreen.intro`)}</p>
      <GreenCheckedItems>
        <p>{t(`dataImportMagicScreen.firstCheck`)}</p>
        <p>{t(`dataImportMagicScreen.secondCheck`, translationContext)}</p>
        <p>{t(`dataImportMagicScreen.thirdCheck`)}</p>
      </GreenCheckedItems>
      <div className='screen__actions'>
        <Button type='button' onClick={() => navigate(`/checklist`)}>
          {t(`button.continue`)}
        </Button>
      </div>
    </>
  );
};

const GreenCheckedItems = ({ children }: { children: ReactNode[] }) => (
  <ul className='usa-icon-list'>
    {children.map((child, index) => (
      <li key={index} className='usa-icon-list__item'>
        <div className='text-green usa-icon-list__icon' aria-hidden='true'>
          <Icon.Check />
        </div>
        <div className='usa-icon-list__content'>{child}</div>
      </li>
    ))}
  </ul>
);
