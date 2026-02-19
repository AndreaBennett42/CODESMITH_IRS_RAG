import { useCallback, useContext, useMemo, useState } from 'react';
import { Grid } from '@trussworks/react-uswds';
import IntroContent from '../IntroContent/IntroContent.js';
import Translation from '../Translation/Translation.js';
import DownloadPDFButton from '../DownloadPDFButton/DownloadPDFButton.js';
import { CURRENT_TAX_YEAR, FEDERAL_RETURN_STATUS } from '../../constants/taxConstants.js';
import { read } from '../../hooks/useApiHook.js';
import { TaxReturnSubmissionStatus } from '../../types/core.js';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';
import { hasBeenSubmitted } from '../../utils/taxReturnUtils.js';

const INITIAL_STATUS_RESPONSE = undefined;
const INITIAL_STATUS_ERROR = undefined;

export const PriorYearTaxReturnCard = () => {
  const { taxReturns, isFetching: isFetchingTaxReturns } = useContext(TaxReturnsContext);

  const [isFetchingPriorYearTaxReturnStatus, setIsFetchingPriorYearTaxReturnStatus] = useState(false);
  const [priorYearTaxReturnStatusFetchError, setPriorYearTaxReturnStatusFetchError] =
    useState<unknown>(INITIAL_STATUS_ERROR);
  const [priorYearTaxReturnStatus, setPriorYearTaxReturnStatus] = useState<TaxReturnSubmissionStatus | undefined>(
    INITIAL_STATUS_RESPONSE
  );

  const fetchStatusByTaxReturnId = useCallback(async (taxReturnId: string) => {
    try {
      setIsFetchingPriorYearTaxReturnStatus(true);
      const statusResponse: TaxReturnSubmissionStatus = await read<TaxReturnSubmissionStatus>(
        `${import.meta.env.VITE_BACKEND_URL}v1/taxreturns/${taxReturnId}/status`
      );
      setPriorYearTaxReturnStatus(statusResponse);
      setPriorYearTaxReturnStatusFetchError(INITIAL_STATUS_ERROR);
    } catch (e) {
      setPriorYearTaxReturnStatusFetchError(e);
      setPriorYearTaxReturnStatus(INITIAL_STATUS_RESPONSE);
    } finally {
      setIsFetchingPriorYearTaxReturnStatus(false);
    }
  }, []);

  const isFetching = isFetchingTaxReturns || isFetchingPriorYearTaxReturnStatus;

  const priorYearTaxReturn = useMemo(() => {
    if (!isFetchingTaxReturns && taxReturns.length > 0) {
      return taxReturns.find((taxReturn) => taxReturn.taxYear === Number.parseInt(CURRENT_TAX_YEAR) - 1);
    }
  }, [taxReturns, isFetchingTaxReturns]);

  const shouldFetchStatus =
    priorYearTaxReturn &&
    hasBeenSubmitted(priorYearTaxReturn) &&
    !isFetchingPriorYearTaxReturnStatus &&
    !priorYearTaxReturnStatusFetchError &&
    !priorYearTaxReturnStatus;

  if (shouldFetchStatus) {
    void fetchStatusByTaxReturnId(priorYearTaxReturn.id);
  }

  if (!isFetching && priorYearTaxReturn && priorYearTaxReturnStatus?.status === FEDERAL_RETURN_STATUS.ACCEPTED) {
    return (
      <Grid
        col={12}
        className='border-base-lighter border-2px shadow-2 margin-top-3'
        data-testid='pastYearTaxReturnCard'
      >
        <h2 className='margin-0  bg-base-lightest padding-2'>
          <Translation i18nKey={`pastYearTaxReturnCard.heading`} collectionId={null}></Translation>
        </h2>
        <div className='text-center padding-205'>
          <DownloadPDFButton taxId={priorYearTaxReturn.id} i18nKey='pastYearTaxReturnCard.button' frontpage={true} />
        </div>
        <div className='padding-left-205 padding-right-205'>
          <h3 className='margin-0'>
            <Translation i18nKey={`pastYearTaxReturnCard.pdfTextTitle`} collectionId={null}></Translation>
          </h3>
          <IntroContent i18nKey={`pastYearTaxReturnCard`} collectionId={null}></IntroContent>
        </div>
      </Grid>
    );
  }
};
