import { Provider } from 'react-redux';
import { mockUseTranslation } from '../../test/mocks/mockFunctions.js';
import { render, screen } from '@testing-library/react';
import { CURRENT_TAX_YEAR, FEDERAL_RETURN_STATUS } from '../../constants/taxConstants.js';
import { FactGraphContextProvider } from '../../factgraph/FactGraphContext.js';
import { TaxReturnCardPostSubmission, TaxReturnPostSubmissionProps } from './TaxReturnCardPostSubmission.js';
import { v4 as uuidv4 } from 'uuid';
import { vi } from 'vitest';
import { wrapComponent } from '../../test/helpers.js';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';
import { StateProfile } from '../../types/StateProfile.js';
import { FetchStateProfileHookResponse } from '../../hooks/useFetchStateProfile.js';
import { setupStore } from '../../redux/store.js';
import {
  SubmissionStatusContext,
  SubmissionStatusContextType,
} from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import { TaxReturn, TaxReturnSubmissionStatus } from '../../types/core.js';

const { mockUseFetchStateProfile } = vi.hoisted(() => {
  return {
    mockUseFetchStateProfile: vi.fn(),
  };
});

vi.mock(`../../hooks/useFetchStateProfile.js`, () => ({
  default: mockUseFetchStateProfile,
}));

vi.mock(`../../hooks/useFact`, () => ({
  default: vi.fn(() => [
    {
      getValue: () => {
        return `ma`;
      },
    },
  ]),
}));

vi.mock(`react-i18next`, () => ({
  useTranslation: mockUseTranslation,
  initReactI18next: {
    type: `3rdParty`,
    init: () => {},
  },
  Trans: ({ children }: never) => children,
}));

describe(`TaxReturnCardPostSubmission`, () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  const taxReturn: TaxReturn = {
    id: uuidv4(),
    taxYear: parseInt(CURRENT_TAX_YEAR),
    facts: {},
    taxReturnSubmissions: [],
    isEditable: true,
    createdAt: ``,
    surveyOptIn: null,
  };

  const defaultProps: TaxReturnPostSubmissionProps = {
    taxReturn,
  };

  const pendingStatus: TaxReturnSubmissionStatus = {
    status: FEDERAL_RETURN_STATUS.PENDING,
    rejectionCodes: [],
    createdAt: ``,
  };

  const errorStatus: TaxReturnSubmissionStatus = {
    status: FEDERAL_RETURN_STATUS.ERROR,
    rejectionCodes: [],
    createdAt: ``,
  };

  const stateProfile: StateProfile = {
    stateCode: `MA`,
    landingUrl: `https://www.irs.gov/`,
    defaultRedirectUrl: ``,
    transferCancelUrl: ``,
    waitingForAcceptanceCancelUrl: ``,
    redirectUrls: [],
    languages: { en: `en`, es: `es` },
    taxSystemName: `FSTSN`,
    acceptedOnly: false,
    customFilingDeadline: null,
    departmentOfRevenueUrl: ``,
    filingRequirementsUrl: ``,
  };

  const fetchStateProfileHookResponse: FetchStateProfileHookResponse = {
    stateProfile: stateProfile,
    isFetching: false,
    fetchError: false,
    fetchSuccess: false,
    fetchSkipped: false,
  };

  type RenderComponentProps = {
    submissionStatusContextOverrides: Partial<SubmissionStatusContextType>;
  };
  const renderComponent = ({ submissionStatusContextOverrides }: RenderComponentProps) => {
    render(
      wrapComponent(
        <Provider store={setupStore()}>
          <TaxReturnsContext.Provider
            value={{
              taxReturns: [],
              currentTaxReturnId: taxReturn.id,
              fetchTaxReturns: vi.fn(),
              isFetching: false,
              fetchSuccess: false,
            }}
          >
            <SubmissionStatusContext.Provider
              value={{
                submissionStatus: pendingStatus,
                setSubmissionStatus: vi.fn(),
                fetchSubmissionStatus: vi.fn(),
                isFetching: false,
                fetchSuccess: true,
                fetchError: false,
                lastFetchAttempt: new Date(),
                ...submissionStatusContextOverrides,
              }}
            >
              <FactGraphContextProvider>
                <TaxReturnCardPostSubmission {...defaultProps} />
              </FactGraphContextProvider>
            </SubmissionStatusContext.Provider>
          </TaxReturnsContext.Provider>
        </Provider>
      )
    );
  };

  it(`should render StateTaxesCard if there is a stateProfile associated with the user's state code`, () => {
    mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponse);
    renderComponent({ submissionStatusContextOverrides: { submissionStatus: pendingStatus } });

    const stateTaxesCardContent = screen.getByTestId(`state-taxes-card`);
    expect(stateTaxesCardContent).toBeDefined();
  });

  it(`should not render StateTaxesCard if the state has no state income tax`, () => {
    mockUseFetchStateProfile.mockReturnValue({});
    renderComponent({ submissionStatusContextOverrides: { submissionStatus: pendingStatus } });

    const stateTaxesCardContent = screen.queryByTestId(`state-taxes-card`);
    expect(stateTaxesCardContent).toBeNull();
  });

  it(`should not render StateTaxesCard if return submission failed`, () => {
    mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponse);
    renderComponent({ submissionStatusContextOverrides: { submissionStatus: errorStatus } });

    const stateTaxesCardContent = screen.queryByTestId(`state-taxes-card`);
    expect(stateTaxesCardContent).toBeNull();
  });

  it.todo(`should render fallback content if status call fails`, () => {
    mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponse);
    renderComponent({
      submissionStatusContextOverrides: {
        submissionStatus: undefined,
        isFetching: false,
        fetchSuccess: false,
        fetchError: true,
        lastFetchAttempt: new Date(),
      },

      // TODO: https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/14764
    });

    const stateTaxesCardContent = screen.queryByTestId(`state-taxes-card`);
    expect(stateTaxesCardContent).toBeNull();
    // TODO: This test fails, but see what actually happens at runtime when status is hardcoded to return e.g a 404
  });
});
