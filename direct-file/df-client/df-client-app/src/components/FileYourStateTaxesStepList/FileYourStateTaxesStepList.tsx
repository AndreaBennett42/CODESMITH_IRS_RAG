import Translation from '../Translation/index.js';
import { FEDERAL_RETURN_STATUS } from '../../constants/taxConstants.js';
import { useContext } from 'react';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import { getCurrentTaxYearReturn } from '../../utils/taxReturnUtils.js';
import useFetchStateProfile from '../../hooks/useFetchStateProfile.js';
import { useTranslation } from 'react-i18next';
import LoadingIndicator from '../LoadingIndicator/LoadingIndicator.js';
import useFact from '../../hooks/useFact.js';
import { Path } from '../../flow/Path.js';

const FileYourStateTaxesStepList = () => {
  const { taxReturns } = useContext(TaxReturnsContext);
  const { submissionStatus } = useContext(SubmissionStatusContext);
  const currentTaxReturn = getCurrentTaxYearReturn(taxReturns);
  const { stateProfile, isFetching } = useFetchStateProfile();
  const [stateCanTransferData] = useFact<boolean>(Path.concretePath(`/stateCanTransferData`, null));

  const { t } = useTranslation();

  const getStepTwoI18nKey = () => {
    // eslint-disable-next-line eqeqeq
    if (stateProfile?.stateCode == `OR`) {
      return `taxReturnCard.fileYourStateTaxesDetails.stepTwoOregon`;
    }

    if (submissionStatus?.status === FEDERAL_RETURN_STATUS.REJECTED) {
      return `taxReturnCard.fileYourStateTaxesDetails.stepTwoRejectedReturn`;
    }

    return `taxReturnCard.fileYourStateTaxesDetails.stepTwoDefault`;
  };

  const showWaitBeforeReturnIsAcceptedMessage = () => {
    if (!stateProfile?.acceptedOnly) {
      return false;
    }

    if (submissionStatus?.status === FEDERAL_RETURN_STATUS.REJECTED) {
      return false;
    }

    if (submissionStatus?.status === FEDERAL_RETURN_STATUS.ACCEPTED) {
      return false;
    }

    return true;
  };

  const context = {
    taxYear: currentTaxReturn?.taxYear,
    stateName: t(`enums.statesAndProvinces.${stateProfile?.stateCode}`),
    stateTaxSystemName: stateProfile?.taxSystemName,
  };

  const stepOneI18nKey =
    submissionStatus?.status === FEDERAL_RETURN_STATUS.REJECTED
      ? `taxReturnCard.fileYourStateTaxesDetails.stepOneRejectedReturn`
      : `taxReturnCard.fileYourStateTaxesDetails.stepOne`;

  if (isFetching) {
    return <LoadingIndicator />;
  }

  if (!stateCanTransferData && submissionStatus?.status === FEDERAL_RETURN_STATUS.REJECTED) {
    return (
      <p>
        <Translation
          i18nKey='taxReturnCard.fileYourStateTaxesDetails.stepOneRejectedReturn'
          collectionId={null}
          context={context}
        />
      </p>
    );
  }

  return (
    <>
      <ol>
        <li>
          <Translation i18nKey={stepOneI18nKey} collectionId={null} context={context} />
        </li>
        <li>
          <Translation i18nKey={getStepTwoI18nKey()} collectionId={null} context={context} />

          {showWaitBeforeReturnIsAcceptedMessage() && (
            <>
              <br />
              <br />
              <Translation i18nKey={`taxReturnCard.fileYourStateTaxesDetails.youWillNeedToWait`} collectionId={null} />
            </>
          )}
        </li>
        <li>
          <Translation
            i18nKey='taxReturnCard.fileYourStateTaxesDetails.stepThree'
            collectionId={null}
            context={context}
          />
        </li>
      </ol>
    </>
  );
};
export default FileYourStateTaxesStepList;
