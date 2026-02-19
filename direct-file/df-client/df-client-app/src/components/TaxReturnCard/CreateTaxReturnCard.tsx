import { useCallback, useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { Button, Grid } from '@trussworks/react-uswds';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import { NetworkConnectionContext } from '../../context/networkConnectionContext.js';
import DFAlert from '../Alert/DFAlert.js';
import { CURRENT_TAX_YEAR } from '../../constants/taxConstants.js';
import { isFlowEnabled } from '../../constants/pageConstants.js';
import { useAppDispatch, useAppSelector } from '../../redux/hooks.js';
import { createTaxReturn, selectCurrentTaxReturnCreate } from '../../redux/slices/tax-return/taxReturnSlice.js';

type Props = {
  setShouldProgressOnSuccessfulTaxReturnCreate: (input: boolean) => void;
};

export const CreateTaxReturnCard = ({ setShouldProgressOnSuccessfulTaxReturnCreate }: Props) => {
  const dispatch = useAppDispatch();
  const { isFetching, hasFetchError } = useAppSelector(selectCurrentTaxReturnCreate);

  const { t } = useTranslation(`translation`);
  const { online } = useContext(NetworkConnectionContext);

  const year = Number.parseInt(CURRENT_TAX_YEAR);

  const createTaxReturnCallback = useCallback(() => {
    dispatch(createTaxReturn());
    setShouldProgressOnSuccessfulTaxReturnCreate(true);
  }, [dispatch, setShouldProgressOnSuccessfulTaxReturnCreate]);

  const disableButton = isFetching || hasFetchError || !online;

  return (
    <Grid col={12} className='border-base-lighter border-2px shadow-2'>
      <h2 className='margin-0  bg-base-lightest padding-2'>{t(`taxReturnCard.header`, { taxYear: year })}</h2>
      <div className='padding-205'>
        {hasFetchError && <DFAlert type='error' i18nKey='home.error' collectionId={null} headingLevel='h3' />}
        {!isFlowEnabled() ? (
          <>
            <div className='margin-bottom-2'>
              <DFAlert type='error' i18nKey='home.endOfTaxSeasonAlert' collectionId={null} headingLevel='h3' />
            </div>
            <div className='text-center'>
              <CommonLinkRenderer className='usa-button' url={`https://www.irs.gov/filing/e-file-options`}>
                {t(`taxReturnCard.otherWaysToFile`)}
              </CommonLinkRenderer>
            </div>
          </>
        ) : (
          <div className='text-center'>
            <Button type='button' onClick={createTaxReturnCallback} disabled={disableButton}>
              {t(`taxReturnCard.notStartedButton`, { taxYear: year })}
            </Button>
          </div>
        )}
      </div>
    </Grid>
  );
};
