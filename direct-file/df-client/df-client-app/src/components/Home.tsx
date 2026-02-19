import { Grid, GridContainer } from '@trussworks/react-uswds';

import PageTitle from './PageTitle/index.js';
import { InProgressTaxReturnCard } from './TaxReturnCard/InProgressTaxReturnCard.js';
import { useContext, useEffect, useState } from 'react';
import TaxReturnCardPostSubmission from './TaxReturnCardPostSubmission/index.js';
import { getTaxReturnById, hasBeenSubmitted } from '../utils/taxReturnUtils.js';
import { BareContentDisplay } from './ContentDisplay/ContentDisplay.js';
import { CreateTaxReturnCard } from './TaxReturnCard/CreateTaxReturnCard.js';
import { TaxProfileContextOrSpinnerGate } from '../screens/TaxProfileContextOrSpinnerGate.js';
import useTranslatePIIRedacted from '../hooks/useTranslatePIIRedacted.js';
import { TaxReturnsContext } from '../context/TaxReturnsContext.js';
import SystemAlertAggregator from './SystemAlertAggregator/SystemAlertAggregator.js';
import SimpleReminderTaxReturnCard from './TaxReturnCard/SimpleReminderTaxReturnCard.js';
import FactAwareReminderTaxReturnCard from './TaxReturnCard/FactAwareReminderTaxReturnCard.js';
import { PriorYearTaxReturnCard } from './TaxReturnCard/PriorYearTaxReturnCard.js';
import { useAppSelector } from '../redux/hooks.js';
import { selectCurrentTaxReturnCreate } from '../redux/slices/tax-return/taxReturnSlice.js';
import { useNavigate } from 'react-router-dom';

const Home = () => {
  const { taxReturns, currentTaxReturnId, isFetching, fetchSuccess } = useContext(TaxReturnsContext);
  const {
    isFetching: isCreating,
    hasFetchError: hasCreateFetchError,
    fetchSuccess: createSuccess,
  } = useAppSelector(selectCurrentTaxReturnCreate);
  const navigate = useNavigate();
  const [shouldProgressOnSuccessfulTaxReturnCreate, setShouldProgressOnSuccessfulTaxReturnCreate] = useState(false);

  const currentTaxReturn = getTaxReturnById(taxReturns, currentTaxReturnId);
  const needsToCreateTaxReturn = fetchSuccess && !currentTaxReturn;
  const isCreatingTaxReturn = isCreating || hasCreateFetchError;
  const shouldRenderCreateTaxReturn = needsToCreateTaxReturn || isCreatingTaxReturn;
  const redacted = useTranslatePIIRedacted(`home.header.body`, true);

  useEffect(() => {
    if (currentTaxReturn && createSuccess && shouldProgressOnSuccessfulTaxReturnCreate) {
      navigate(`/pre-checklist`, { state: { firstVisit: true } });
    }
  }, [currentTaxReturn, createSuccess, navigate, shouldProgressOnSuccessfulTaxReturnCreate]);

  return (
    <>
      <PageTitle redactedTitle={redacted} large>
        <BareContentDisplay i18nKey='home.header' />
      </PageTitle>
      <SystemAlertAggregator />
      {(isFetching || fetchSuccess) && (
        <GridContainer className='margin-top-5'>
          <Grid row>
            {shouldRenderCreateTaxReturn && (
              <>
                <CreateTaxReturnCard
                  setShouldProgressOnSuccessfulTaxReturnCreate={setShouldProgressOnSuccessfulTaxReturnCreate}
                />
                <SimpleReminderTaxReturnCard />
                <PriorYearTaxReturnCard />
              </>
            )}
            {currentTaxReturn ? (
              <TaxProfileContextOrSpinnerGate>
                {hasBeenSubmitted(currentTaxReturn) ? (
                  // TODO: Support multiple returns on home page.
                  //       We will need a new utility to be able to simultaneously load facts for each post submission
                  //       card since they rely on useFact which is only supported by one FactGraphContext instance
                  //       today
                  //       https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/5105
                  <>
                    <TaxReturnCardPostSubmission taxReturn={currentTaxReturn} />
                    <PriorYearTaxReturnCard />
                  </>
                ) : (
                  <>
                    <InProgressTaxReturnCard taxReturn={currentTaxReturn} />
                    <FactAwareReminderTaxReturnCard />
                    <PriorYearTaxReturnCard />
                  </>
                )}
              </TaxProfileContextOrSpinnerGate>
            ) : null}
          </Grid>
        </GridContainer>
      )}
    </>
  );
};

export default Home;
