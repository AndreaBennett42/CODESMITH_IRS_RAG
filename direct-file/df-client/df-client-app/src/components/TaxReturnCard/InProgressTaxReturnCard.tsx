import { FC, useCallback, useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { Button, Grid } from '@trussworks/react-uswds';
import { NetworkConnectionContext } from '../../context/networkConnectionContext.js';
import DFAlert from '../Alert/DFAlert.js';
import { TaxReturn } from '../../types/core.js';
import Translation from '../Translation/index.js';
import useFact from '../../hooks/useFact.js';
import { Path } from '../../flow/Path.js';
// eslint-disable-next-line max-len
import PaperPathStatusAlert from '../PaperPathStatusAlert/PaperPathStatusAlert.js';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import { isFlowEnabled } from '../../constants/pageConstants.js';

export interface InProgressTaxReturnCardProps {
  taxReturn: TaxReturn;
}

export const InProgressTaxReturnCard: FC<InProgressTaxReturnCardProps> = ({ taxReturn }) => {
  const { t } = useTranslation(`translation`);
  const navigate = useNavigate();
  const { online } = useContext(NetworkConnectionContext);

  const [isPaperPath] = useFact<boolean>(Path.concretePath(`/isPaperPath`, null));

  const handleClick = useCallback(async () => {
    navigate(`/checklist`);
  }, [navigate]);

  const disableButton = !online;

  return (
    <Grid col={12} className='border-base-lighter border-2px shadow-2'>
      <h2 className='margin-0  bg-base-lightest padding-2'>
        {t(`taxReturnCard.header`, taxReturn as { taxYear: number })}
      </h2>
      <div className='padding-205'>
        {/* if flow is disabled render error otherwise render paperPath Alert or regular inProgress status */}
        {!isFlowEnabled() ? (
          <DFAlert
            type='error'
            showTextAsHeader
            i18nKey='home.endOfTaxSeasonAlert'
            collectionId={null}
            headingLevel='h3'
          />
        ) : isPaperPath ? (
          <PaperPathStatusAlert />
        ) : (
          <DFAlert
            type='info'
            showTextAsHeader
            i18nKey={`taxReturnCard.status.inProgress`}
            collectionId={null}
            headingLevel='h3'
          />
        )}
        <p className='usa-prose'>
          <Translation i18nKey='taxReturnCard.taxReturnId' collectionId={null} context={taxReturn} />
        </p>
        {!isFlowEnabled() ? (
          <>
            <hr className='margin-y-3' />
            <div className='usa-prose'>
              <p>
                <Translation i18nKey='taxReturnCard.whatToDoNext' collectionId={null} />
              </p>
              <p>
                <Translation i18nKey='taxReturnCard.otherWaysToFilePreamble' collectionId={null} />
              </p>
            </div>
            <div className='text-center margin-top-2'>
              <CommonLinkRenderer className='usa-button' url={`https://www.irs.gov/filing/e-file-options`}>
                {t(`taxReturnCard.otherWaysToFile`)}
              </CommonLinkRenderer>
            </div>
          </>
        ) : (
          <div className='text-center'>
            <Button type='button' onClick={handleClick} disabled={disableButton}>
              {t(`taxReturnCard.inProgressButton`, taxReturn as { taxYear: number })}
            </Button>
          </div>
        )}
      </div>
    </Grid>
  );
};
