import LoadingIndicator from '../LoadingIndicator/LoadingIndicator.js';
import { Path } from '../../flow/Path.js';
import useFact from '../../hooks/useFact.js';
import useFetchStateProfile from '../../hooks/useFetchStateProfile.js';
import { CURRENT_TAX_YEAR, FEDERAL_RETURN_STATUS } from '../../constants/taxConstants.js';
import { useContext } from 'react';
import StateTaxReminderAlert from '../TaxReturnCardPostSubmission/StateTaxReminderAlert/StateTaxReminderAlert.js';
import { getTaxReturnById } from '../../utils/taxReturnUtils.js';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';

const StateTaxReminderAlertWrapper = () => {
  const { stateProfile, isFetching } = useFetchStateProfile();
  const { submissionStatus, isFetching: isFetchingStatus } = useContext(SubmissionStatusContext);

  const [scopedStateDoesNotHavePersonalIncomeTax] = useFact<boolean>(
    Path.concretePath(`/scopedStateDoesNotHavePersonalIncomeTax`, null)
  );

  const { taxReturns, currentTaxReturnId } = useContext(TaxReturnsContext);
  const currentTaxReturn = getTaxReturnById(taxReturns, currentTaxReturnId);
  const taxYear = currentTaxReturn?.taxYear ? currentTaxReturn.taxYear : parseInt(CURRENT_TAX_YEAR);

  if (isFetching || isFetchingStatus) {
    return <LoadingIndicator />;
  }

  // do not display any alert message on the submission confirmation page if a user has
  // navigated back to it after their return was rejected
  if (submissionStatus?.status === FEDERAL_RETURN_STATUS.REJECTED) {
    return;
  }

  if (stateProfile) {
    return (
      <StateTaxReminderAlert
        stateProfile={stateProfile}
        taxYear={taxYear}
        scopedStateDoesNotHavePersonalIncomeTax={scopedStateDoesNotHavePersonalIncomeTax ?? false}
      />
    );
  }
};

export default StateTaxReminderAlertWrapper;
