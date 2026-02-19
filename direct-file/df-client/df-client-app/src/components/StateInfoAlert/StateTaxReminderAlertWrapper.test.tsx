import { vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TaxReturn, TaxReturnSubmissionStatus } from '../../types/core.js';
import { wrapComponent } from '../../test/helpers.js';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import { v4 as uuidv4 } from 'uuid';
import { CURRENT_TAX_YEAR, FEDERAL_RETURN_STATUS } from '../../constants/taxConstants.js';
import StateTaxReminderAlertWrapper from './StateTaxReminderAlertWrapper.js';
import { StateProfile } from '../../types/StateProfile.js';
import { FetchStateProfileHookResponse } from '../../hooks/useFetchStateProfile.js';
import { ConcretePath } from '@irs/js-factgraph-scala';

const { mockUseFetchStateProfile } = vi.hoisted(() => {
  return {
    mockUseFetchStateProfile: vi.fn(),
  };
});

vi.mock(`../../hooks/useFetchStateProfile.js`, () => ({
  default: mockUseFetchStateProfile,
}));

const { t } = vi.hoisted(() => {
  return {
    t: vi.fn((key: string) => key),
  };
});

vi.mock(`react-i18next`, () => ({
  useTranslation: () => {
    return {
      t: t,
      i18n: {
        language: `en`,
        exists: vi.fn(() => true),
      },
    };
  },
  initReactI18next: {
    type: `3rdParty`,
  },
  Trans: ({ i18nKey }: { i18nKey: string }) => t(i18nKey),
}));

const { useFact } = vi.hoisted(() => {
  return {
    useFact: vi.fn((_path: ConcretePath) => [false]),
  };
});

vi.mock(`../../hooks/useFact`, () => ({
  default: useFact,
}));

describe(`StateTaxReminderAlertWrapper`, () => {
  const defaultTaxReturn: TaxReturn = {
    id: uuidv4(),
    facts: {},
    taxYear: parseInt(CURRENT_TAX_YEAR),
    taxReturnSubmissions: [],
    isEditable: true,
    createdAt: ``,
    surveyOptIn: null,
  };

  const pendingStatus: TaxReturnSubmissionStatus = {
    status: FEDERAL_RETURN_STATUS.PENDING,
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

  const renderComponent = (status: TaxReturnSubmissionStatus = pendingStatus) => {
    render(
      wrapComponent(
        <TaxReturnsContext.Provider
          value={{
            taxReturns: [defaultTaxReturn],
            currentTaxReturnId: `foo`,
            fetchTaxReturns: vi.fn(),
            isFetching: false,
            fetchSuccess: true,
          }}
        >
          <SubmissionStatusContext.Provider
            value={{
              submissionStatus: status,
              setSubmissionStatus: vi.fn(),
              fetchSubmissionStatus: vi.fn(),
              isFetching: false,
              fetchSuccess: true,
              fetchError: false,
              lastFetchAttempt: new Date(),
            }}
          >
            <StateTaxReminderAlertWrapper />
          </SubmissionStatusContext.Provider>
        </TaxReturnsContext.Provider>
      )
    );

    const link = screen.queryByRole(`link`);

    return { link };
  };

  it(`should render link to internal file our state taxes page if the state is integrated`, () => {
    mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponse);
    useFact.mockImplementation((path: ConcretePath) => {
      if (path.includes(`/stateCanTransferData`)) {
        return [true];
      }
      return [false];
    });

    const { link } = renderComponent();

    expect(link).toHaveTextContent(`taxReturnCard.stateFilingReminder.fileYourStateTaxesWithTransfer.linkText`);

    expect(link).toHaveProperty(`href`, `http://localhost:3000/file-your-state-taxes`);
  });

  it(`should render link to external state tax website if state cannot transfer data`, () => {
    mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponse);

    useFact.mockImplementation((path: ConcretePath) => {
      if (path.includes(`/stateCanTransferData`)) {
        return [false];
      }

      return [false];
    });

    const { link } = renderComponent();

    expect(link).toHaveTextContent(`taxReturnCard.stateFilingReminder.fileYourStateTaxesWithoutTransfer.linkText`);

    expect(link).toHaveProperty(`href`, `${stateProfile.landingUrl}?ref_location=df_submission`);
  });

  it(`should render link to washing state tax website if state state is washington`, () => {
    const washingStateProfile = {
      ...fetchStateProfileHookResponse,
      stateProfile: {
        ...stateProfile,
        stateCode: `WA`,
      },
    };

    mockUseFetchStateProfile.mockReturnValue(washingStateProfile);

    useFact.mockImplementation((path: ConcretePath) => {
      if (path.includes(`/stateCanTransferData`)) {
        return [false];
      }

      return [false];
    });

    const { link } = renderComponent();

    expect(link).toHaveTextContent(`taxReturnCard.stateFilingReminder.washington.linkText`);

    expect(link).toHaveProperty(`href`, `${stateProfile.landingUrl}?ref_location=df_submission`);
  });

  it(`should not render alert if state has no state tax income (as indicated by the lack of a state profile)`, () => {
    mockUseFetchStateProfile.mockReturnValue({});

    useFact.mockImplementation((path: ConcretePath) => {
      if (path.includes(`/stateCanTransferData`)) {
        return [false];
      }

      return [false];
    });

    const { link } = renderComponent();
    expect(link).toBeNull();
  });
});
