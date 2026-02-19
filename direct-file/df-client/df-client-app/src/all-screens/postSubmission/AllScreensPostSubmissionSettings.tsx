import {
  ChangeEventHandler,
  createContext,
  Dispatch,
  ReactNode,
  SetStateAction,
  useCallback,
  useContext,
  useState,
} from 'react';
import { federalReturnStatuses } from '../../constants/taxConstants.js';
import styles from '../AllScreens.module.scss';
import { Checkbox, Label, Select, TextInput } from '@trussworks/react-uswds';
import { FederalReturnStatus, RejectedStatus, TaxReturn } from '../../types/core.js';
import { StateOrProvince, statesAndProvinces } from '../../types/StateOrProvince.js';
import {
  DEFAULT_FETCH_STATE_PROFILE_HOOK_RESPONSE,
  DEFAULT_SUBMISSION_STATUS_CONTEXT,
  DEFAULT_TAX_RETURNS_SLICE_STATE_DATA,
  DEFAULT_TRANSFER_DISABLED,
  FetchStateProfileHookResponseStateProfileNotOptional,
  SubmissionStatusContextTypeStatusNotOptional,
} from './stateTaxesSettingsDefaults.js';
import { selectCurrentTaxReturn, TaxReturnsSliceStateData } from '../../redux/slices/tax-return/taxReturnSlice.js';
import { useTranslation } from 'react-i18next';
import { MEF_REJECTION_ERROR_CODES } from '../../constants/rejectionConstants.js';

export type AllScreensStateTaxesSettingsContextType = {
  fetchStateProfileHookResponse: FetchStateProfileHookResponseStateProfileNotOptional;
  setFetchStateProfileHookResponse: Dispatch<SetStateAction<FetchStateProfileHookResponseStateProfileNotOptional>>;
  currentTaxReturn: TaxReturn | undefined;
  hasTaxReturnsFetchError: boolean;
  isFetchingTaxReturns: boolean;
  setTaxReturnsSliceStateData: Dispatch<SetStateAction<TaxReturnsSliceStateData>>;
  submissionStatusContext: SubmissionStatusContextTypeStatusNotOptional;
  setSubmissionStatusContext: Dispatch<SetStateAction<SubmissionStatusContextTypeStatusNotOptional>>;
  transferDisabled: boolean;
  setTransferDisabled: Dispatch<SetStateAction<boolean>>;
};
export const AllScreensStateTaxesSettingsContext = createContext<AllScreensStateTaxesSettingsContextType | undefined>(
  undefined
);

export const useAllScreensStateTaxesSettingsContext = (): AllScreensStateTaxesSettingsContextType => {
  const allScreensStateTaxesSettingsContext = useContext(AllScreensStateTaxesSettingsContext);
  if (allScreensStateTaxesSettingsContext === undefined) {
    throw new Error(
      `useAllScreensStateTaxesSettingsContext must be used within a corresponding context Provider with a defined value`
    );
  }

  return allScreensStateTaxesSettingsContext;
};

type AllScreensStateTaxesSettingsContextProviderProps = {
  children: ReactNode;
};
export const AllScreensStateTaxesSettingsContextProvider = ({
  children,
}: AllScreensStateTaxesSettingsContextProviderProps) => {
  const [fetchStateProfileHookResponse, setFetchStateProfileHookResponse] =
    useState<FetchStateProfileHookResponseStateProfileNotOptional>(DEFAULT_FETCH_STATE_PROFILE_HOOK_RESPONSE);
  const [taxReturnsSliceStateData, setTaxReturnsSliceStateData] = useState<TaxReturnsSliceStateData>(
    DEFAULT_TAX_RETURNS_SLICE_STATE_DATA
  );
  const [submissionStatusContext, setSubmissionStatusContext] = useState<SubmissionStatusContextTypeStatusNotOptional>({
    ...DEFAULT_SUBMISSION_STATUS_CONTEXT,
    fetchSubmissionStatus: () => mockFetchSubmissionStatus(),
  });
  const mockFetchSubmissionStatus = () =>
    setSubmissionStatusContext((previousSubmissionStatusContext) => ({
      ...previousSubmissionStatusContext,
      lastFetchAttempt: new Date(),
    }));
  const [transferDisabled, setTransferDisabled] = useState<boolean>(DEFAULT_TRANSFER_DISABLED);
  const currentTaxReturn = selectCurrentTaxReturn({ taxReturns: { data: taxReturnsSliceStateData } });
  const hasTaxReturnsFetchError = taxReturnsSliceStateData.hasFetchError;
  const isFetchingTaxReturns = taxReturnsSliceStateData.isFetching;

  return (
    <AllScreensStateTaxesSettingsContext.Provider
      value={{
        fetchStateProfileHookResponse,
        setFetchStateProfileHookResponse,
        hasTaxReturnsFetchError,
        isFetchingTaxReturns,
        currentTaxReturn,
        setTaxReturnsSliceStateData,
        submissionStatusContext,
        setSubmissionStatusContext,
        transferDisabled,
        setTransferDisabled,
      }}
    >
      {children}
    </AllScreensStateTaxesSettingsContext.Provider>
  );
};

export const AllScreensStateTaxesSettings = () => {
  const {
    fetchStateProfileHookResponse,
    setFetchStateProfileHookResponse,
    submissionStatusContext,
    setSubmissionStatusContext,
    transferDisabled,
    setTransferDisabled,
  } = useAllScreensStateTaxesSettingsContext();
  const { t } = useTranslation();

  const handleTransferDisabledChange: ChangeEventHandler<HTMLInputElement> = useCallback(
    (e) => {
      setTransferDisabled(!e.target.checked);
    },
    [setTransferDisabled]
  );

  const handleStatusSettingChange: ChangeEventHandler<HTMLSelectElement> = useCallback(
    (e) => {
      setSubmissionStatusContext((prevSubmissionStatusContext) => ({
        ...prevSubmissionStatusContext,
        submissionStatus: {
          ...prevSubmissionStatusContext.submissionStatus,
          status: e.target.value as FederalReturnStatus,
        },
      }));
    },
    [setSubmissionStatusContext]
  );

  const handleStateCodeChange: ChangeEventHandler<HTMLSelectElement> = useCallback(
    (e) => {
      setFetchStateProfileHookResponse((prevFetchStateProfileHookResponse) => ({
        ...prevFetchStateProfileHookResponse,
        stateProfile: {
          ...prevFetchStateProfileHookResponse.stateProfile,
          stateCode: e.target.value as StateOrProvince,
        },
      }));
    },
    [setFetchStateProfileHookResponse]
  );

  const handleTaxSystemNameChange: ChangeEventHandler<HTMLInputElement> = useCallback(
    (e) => {
      setFetchStateProfileHookResponse((prevFetchStateProfileHookResponse) => ({
        ...prevFetchStateProfileHookResponse,
        stateProfile: {
          ...prevFetchStateProfileHookResponse.stateProfile,
          taxSystemName: e.target.value,
        },
      }));
    },
    [setFetchStateProfileHookResponse]
  );

  const handleAcceptedOnlyChange: ChangeEventHandler<HTMLInputElement> = useCallback(
    (e) => {
      setFetchStateProfileHookResponse((prevFetchStateProfileHookResponse) => ({
        ...prevFetchStateProfileHookResponse,
        stateProfile: {
          ...prevFetchStateProfileHookResponse.stateProfile,
          acceptedOnly: e.target.checked,
        },
      }));
    },
    [setFetchStateProfileHookResponse]
  );

  const [selectedMefCodes, setSelectedMefCodes] = useState<RejectedStatus[]>([]);
  const handleMefCodeChange: ChangeEventHandler<HTMLInputElement> = useCallback(
    (e) => {
      if (e.target.checked) {
        const newCodes = [
          {
            MeFErrorCode: e.target.value,
            TranslationKey: ``,
            MeFDescription: ``,
          },
          ...selectedMefCodes,
        ];
        setSelectedMefCodes(newCodes);
        setSubmissionStatusContext((prevSubmissionStatusContext) => ({
          ...prevSubmissionStatusContext,
          submissionStatus: {
            ...prevSubmissionStatusContext.submissionStatus,
            rejectionCodes: newCodes,
          },
        }));
      } else {
        const newCodes = selectedMefCodes.filter((code) => e.target.value !== code.MeFErrorCode);
        setSelectedMefCodes(newCodes);
        setSubmissionStatusContext((prevSubmissionStatusContext) => ({
          ...prevSubmissionStatusContext,
          submissionStatus: {
            ...prevSubmissionStatusContext.submissionStatus,
            rejectionCodes: newCodes,
          },
        }));
      }
    },
    [selectedMefCodes, setSubmissionStatusContext]
  );
  const fixableMefCodesInTranslation = t(`taxReturnDetails.rejectedReturnDetails.fixable.singleErrorExplanation`, {
    returnObjects: true,
  });
  const mefCodes = [...Object.keys(fixableMefCodesInTranslation), ...MEF_REJECTION_ERROR_CODES.UNFIXABLE_BY_DF];

  return (
    <>
      <div className={styles.flyoutMenuSection}>
        <fieldset className={`usa-fieldset ${styles.flyoutMenuSection}`}>
          <legend>System</legend>
          <Checkbox
            id='return-transfer-is-enabled'
            name='return-transfer-is-enabled'
            label='return transfer is enabled'
            checked={!transferDisabled}
            onChange={handleTransferDisabledChange}
          />
        </fieldset>
      </div>
      <div className={styles.flyoutMenuSection}>
        <fieldset className={`usa-fieldset ${styles.flyoutMenuSection}`}>
          <legend>Status</legend>
          <Select
            id='federal-return-status'
            name='federal-return-status'
            value={submissionStatusContext.submissionStatus.status}
            onChange={handleStatusSettingChange}
          >
            {federalReturnStatuses.map((status, index) => (
              <option key={`${index}.${status}`} value={status}>
                {status}
              </option>
            ))}
          </Select>
        </fieldset>
      </div>
      <fieldset className={`usa-fieldset overflow-y-scroll height-card ${styles.flyoutMenuSection} `}>
        <legend>MeF Rejection Code</legend>
        {mefCodes.map((code, index) => {
          return (
            <Checkbox
              disabled={submissionStatusContext.submissionStatus.status !== `Rejected`}
              key={`mef-code-${index}`}
              id={`mef-code-${code}`}
              name={`mef-code-${code}`}
              label={code}
              value={code}
              checked={selectedMefCodes.map((c) => c.MeFErrorCode).includes(code)}
              onChange={handleMefCodeChange}
            />
          );
        })}
      </fieldset>
      <div className={styles.flyoutMenuSection}>
        <fieldset className={`usa-fieldset ${styles.flyoutMenuSection}`}>
          <legend>State Profile</legend>
          <Label htmlFor='state-profile-state-code'>stateCode</Label>
          <Select
            id='state-profile-state-code'
            name='state-profile-state-code'
            value={fetchStateProfileHookResponse.stateProfile.stateCode}
            onChange={handleStateCodeChange}
          >
            {statesAndProvinces.map((code, index) => (
              <option key={`${index}.${code}`} value={code}>
                {code}
              </option>
            ))}
          </Select>
          <Label htmlFor='state-profile-tax-system-name'>taxSystemName</Label>
          <TextInput
            id='state-profile-tax-system-name'
            name='state-profile-tax-system-name'
            type='text'
            value={fetchStateProfileHookResponse.stateProfile.taxSystemName}
            onChange={handleTaxSystemNameChange}
            spellCheck={false}
          />
          <Checkbox
            id='state-profile-accepted-only'
            name='state-profile-accepted-only'
            label='acceptedOnly'
            checked={fetchStateProfileHookResponse.stateProfile.acceptedOnly}
            onChange={handleAcceptedOnlyChange}
          />
        </fieldset>
      </div>
    </>
  );
};
