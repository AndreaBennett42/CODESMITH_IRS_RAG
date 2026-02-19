import { Link } from '@trussworks/react-uswds';
import { useContext, useState } from 'react';
import { hasBeenSubmitted } from '../../utils/taxReturnUtils.js';
import { useNavigate, useSearchParams } from 'react-router-dom';
import TransferReturnScreen from './TransferReturnScreen/TransferReturnScreen.js';
import WaitingForAcceptanceScreen from './WaitingForAcceptanceScreen/WaitingForAcceptanceScreen.js';
import RejectedReturnScreen from './ReturnRejectedScreen/ReturnRejectedScreen.js';
import { FEDERAL_RETURN_STATUS } from '../../constants/taxConstants.js';
import { StateProfile } from '../../types/StateProfile.js';
import { Trans, useTranslation } from 'react-i18next';
import { formatAndAppendHeaders } from '../../misc/apiHelpers.js';
import { isReturnTransferEnabled, REF_LOCATION, REF_LOCATION_VALUE } from '../../constants/pageConstants.js';
import ErrorScreen from './ErrorScreen/ErrorScreen.js';
import ReturnErrorScreen from './ReturnErrorScreen/ReturnErrorScreen.js';
import LoadingIndicator from '../../components/LoadingIndicator/LoadingIndicator.js';
import ReturnTransferDisabledScreen from './ReturnTransferDisabledScreen/ReturnTransferDisabledScreen.js';
import { TaxProfileContextOrSpinnerGate } from '../TaxProfileContextOrSpinnerGate.js';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import useFetchStateProfile, { FetchStateProfileHookResponse } from '../../hooks/useFetchStateProfile.js';
import { useAppSelector } from '../../redux/hooks.js';
import { selectCurrentTaxReturn } from '../../redux/slices/tax-return/taxReturnSlice.js';
import { TaxReturn } from '../../types/core.js';
import { StateApiErrorCode } from '../../constants/stateApiConstants.js';
import {
  SetSystemAlertConfig,
  SystemAlertKey,
  useSystemAlertContext,
} from '../../context/SystemAlertContext/SystemAlertContext.js';

type CreateAuthorizationCodeResponse = {
  authorizationCode?: string;
  error?: string;
};

export const AUTHORIZATION_CODE_PARAM_NAME = `authorizationCode`;
export const SESSION_ID_PARAM_NAME = `session-id`;
export const REDIRECT_PARAM_NAME = `redirect`;

const appendQueryParams = (url: URL, sessionIdParam: string | null) => {
  url.searchParams.append(REF_LOCATION, REF_LOCATION_VALUE.AUTHSTATE);
  if (sessionIdParam) {
    url.searchParams.append(SESSION_ID_PARAM_NAME, sessionIdParam);
  }
};

export const determineRedirectUrl = (
  defaultRedirectUrl: string,
  redirectParam: string | null,
  allowedRedirectUrls: StateProfile['redirectUrls'],
  authorizationCode: string,
  sessionIdParam: string | null
) => {
  const baseUrl = redirectParam && allowedRedirectUrls.includes(redirectParam) ? redirectParam : defaultRedirectUrl;
  const redirectUrl = new URL(baseUrl);

  redirectUrl.searchParams.append(AUTHORIZATION_CODE_PARAM_NAME, authorizationCode);

  appendQueryParams(redirectUrl, sessionIdParam);

  return redirectUrl;
};

const AuthorizeStateScreen = () => {
  const currentTaxReturnId = useAppSelector((store) => store.taxReturns.data.currentTaxReturnId);

  return currentTaxReturnId ? (
    <TaxProfileContextOrSpinnerGate>
      <AuthorizeStateScreenContextRenderer />
    </TaxProfileContextOrSpinnerGate>
  ) : (
    <AuthorizeStateScreenContextRenderer />
  );
};

const AuthorizeStateScreenContextRenderer = () => {
  const fetchStateProfileHookResponse = useFetchStateProfile();
  const transferEnabled = isReturnTransferEnabled();

  const isFetchingTaxReturns = useAppSelector((store) => store.taxReturns.data.isFetching);
  const hasTaxReturnsFetchError = useAppSelector((store) => store.taxReturns.data.hasFetchError);
  const currentTaxReturn = useAppSelector(selectCurrentTaxReturn);

  return (
    <AuthorizeStateScreenContent
      currentTaxReturn={currentTaxReturn}
      isFetchingTaxReturns={isFetchingTaxReturns}
      hasTaxReturnsFetchError={hasTaxReturnsFetchError}
      fetchStateProfileHookResponse={fetchStateProfileHookResponse}
      transferDisabled={!transferEnabled}
    />
  );
};

export type AuthorizeStateScreenContentProps = {
  currentTaxReturn: TaxReturn | undefined;
  isFetchingTaxReturns: boolean;
  hasTaxReturnsFetchError: boolean;
  fetchStateProfileHookResponse: FetchStateProfileHookResponse;
  transferDisabled: boolean;
};

export const AuthorizeStateScreenContent = ({
  currentTaxReturn,
  isFetchingTaxReturns,
  hasTaxReturnsFetchError,
  fetchStateProfileHookResponse,
  transferDisabled,
}: AuthorizeStateScreenContentProps) => {
  const { t } = useTranslation(`translation`, { keyPrefix: `authorizeState` });
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const [isSubmittingTransfer, setIsSubmittingTransfer] = useState(false);

  const { setSystemAlert, deleteSystemAlert } = useSystemAlertContext();

  const {
    submissionStatus,
    isFetching: fetchingStatus,
    fetchError: statusFetchError,
    fetchSubmissionStatus,
  } = useContext(SubmissionStatusContext);
  const {
    stateProfile,
    isFetching: fetchingStateProfile,
    fetchSuccess: stateProfileFetchSuccess,
    fetchError: stateProfileFetchError,
    fetchSkipped: stateProfileFetchSkipped,
  } = fetchStateProfileHookResponse;

  // This page relies on several sequential api calls.
  const hasAttemptedToFetchStateProfile =
    stateProfileFetchSkipped || stateProfileFetchSuccess || stateProfileFetchError;
  const isLoading =
    // if any call is fetching
    isFetchingTaxReturns ||
    fetchingStatus ||
    fetchingStateProfile ||
    // OR none have failed, BUT we have not yet started to fetch the state profile
    // (prevents a split-second render where the state profile is null)
    (!hasTaxReturnsFetchError && !statusFetchError && !hasAttemptedToFetchStateProfile);

  const goHome = () => navigate(`/home`);

  if (isLoading) {
    return <LoadingIndicator delayMS={0} />;
  }

  const hasSubmittedTaxReturn = currentTaxReturn && hasBeenSubmitted(currentTaxReturn);
  if (!hasSubmittedTaxReturn) {
    // Taxpayer is here in error.
    // Taxpayer *must* have submitted a tax return using Direct File and have been redirected from a state
    // This could happen if they log in to the wrong account, or if someone else's Direct File session is still active
    // on the machine they're using
    return <ErrorScreen errorMessage={t(`errors.returnNotSubmitted`)} handleGoBack={goHome} />;
  }

  if (!stateProfile || stateProfile.defaultRedirectUrl === null) {
    // Fail fast, we need to be able to determine the state and get their state profile,
    // otherwise nothing else makes sense to do
    // Additionally, if the state does not have a defaultRedirectUrl, they are not configured for data transfer.
    // taxpayer is here in error or system is misconfigured.
    return <ErrorScreen errorMessage={t(`errors.stateProfileNotFound`)} handleGoBack={goHome} />;
  }

  const { id: taxReturnUuid, taxYear } = currentTaxReturn;

  if (transferDisabled) {
    return <ReturnTransferDisabledScreen taxReturnUuid={currentTaxReturn.id} />;
  }

  const {
    stateCode,
    taxSystemName,
    landingUrl,
    defaultRedirectUrl,
    transferCancelUrl,
    waitingForAcceptanceCancelUrl,
    redirectUrls,
    acceptedOnly: federalReturnMustBeAccepted,
  } = stateProfile;

  const federalReturnStatus = submissionStatus?.status;

  if (!federalReturnStatus) {
    return (
      <ErrorScreen
        errorMessage={<Trans t={t} i18nKey={`errors.unableToDetermineTaxReturnStatus`} />}
        handleGoBack={goHome}
      />
    );
  }

  const isPending = federalReturnStatus === FEDERAL_RETURN_STATUS.PENDING;
  const isAccepted = federalReturnStatus === FEDERAL_RETURN_STATUS.ACCEPTED;
  const isRejected = federalReturnStatus === FEDERAL_RETURN_STATUS.REJECTED;
  const isError = federalReturnStatus === FEDERAL_RETURN_STATUS.ERROR;
  const isReadyToTransfer = isAccepted || (!federalReturnMustBeAccepted && isPending);
  const isWaitingForAcceptance = federalReturnMustBeAccepted && isPending;

  const sessionId = searchParams.get(SESSION_ID_PARAM_NAME);

  if (isReadyToTransfer) {
    const handleSubmit = async () => {
      setIsSubmittingTransfer(true);
      const handleGenerateAuthorizationCodeErrors = (error: unknown) => {
        if (error === StateApiErrorCode.E_TAX_RETURN_NOT_ACCEPTED_OR_PENDING) {
          // The user might have a pending return on page load, which gets rejected
          // before they click the transfer button.
          // In that case, re-fetch status to show them their (likely) Rejected status.
          fetchSubmissionStatus(taxReturnUuid);
        }

        const config: SetSystemAlertConfig = {
          type: `error`,
          i18nKey: `stateTransfer.unableToGenerateAuthorizationCode`,
          context: {
            errorCode: error,
          },
        };
        setSystemAlert(SystemAlertKey.CREATE_AUTHORIZATION_CODE, config);
      };

      try {
        const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}v1/state-api/authorization-code`, {
          method: `POST`,
          headers: formatAndAppendHeaders({}),
          body: JSON.stringify({
            taxReturnUuid,
            taxYear,
          }),
        });

        const responseBody: CreateAuthorizationCodeResponse = await response.json();
        setIsSubmittingTransfer(false);

        if (response.ok && responseBody.authorizationCode) {
          deleteSystemAlert(SystemAlertKey.CREATE_AUTHORIZATION_CODE);

          const customRedirectUrl = searchParams.get(REDIRECT_PARAM_NAME);

          const redirectUrl = determineRedirectUrl(
            defaultRedirectUrl,
            customRedirectUrl,
            redirectUrls,
            responseBody.authorizationCode,
            sessionId
          );
          window.location.assign(redirectUrl);
        } else {
          handleGenerateAuthorizationCodeErrors(responseBody.error);
        }
      } catch (error) {
        handleGenerateAuthorizationCodeErrors(error);
      }
    };

    const cancelUrl = new URL(transferCancelUrl || landingUrl);
    appendQueryParams(cancelUrl, sessionId);

    return (
      <TransferReturnScreen
        taxYear={taxYear}
        taxReturnStatus={federalReturnStatus}
        taxReturnUuid={taxReturnUuid}
        stateCode={stateCode}
        stateTaxSystemName={taxSystemName}
        handleSubmit={handleSubmit}
        goBackUrl={cancelUrl}
        isSubmittingTransfer={isSubmittingTransfer}
      />
    );
  }

  if (isWaitingForAcceptance) {
    const cancelUrl = new URL(waitingForAcceptanceCancelUrl || landingUrl);
    appendQueryParams(cancelUrl, sessionId);

    return (
      <WaitingForAcceptanceScreen
        taxYear={taxYear}
        taxReturnStatus={federalReturnStatus}
        stateCode={stateCode}
        stateTaxSystemName={taxSystemName}
        goBackUrl={cancelUrl}
        taxReturnId={currentTaxReturn.id}
      />
    );
  }

  if (isRejected) {
    return (
      <RejectedReturnScreen
        taxYear={taxYear}
        taxReturnStatus={federalReturnStatus}
        stateCode={stateCode}
        stateTaxSystemName={taxSystemName}
        federalReturnMustBeAccepted={federalReturnMustBeAccepted}
        handleGoBack={goHome}
      />
    );
  }

  if (isError) {
    return <ReturnErrorScreen taxYear={taxYear} />;
  }
};

export default AuthorizeStateScreen;
