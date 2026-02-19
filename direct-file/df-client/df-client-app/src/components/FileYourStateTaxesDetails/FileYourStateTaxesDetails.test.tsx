import { render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import { wrapComponent } from '../../test/helpers.js';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import FileYourStateTaxesDetails from './FileYourStateTaxesDetails.js';
import { TaxReturn, TaxReturnSubmissionStatus } from '../../types/core.js';
import { v4 as uuidv4 } from 'uuid';
import { CURRENT_TAX_YEAR, FEDERAL_RETURN_STATUS } from '../../constants/taxConstants.js';
import { FetchStateProfileHookResponse } from '../../hooks/useFetchStateProfile.js';
import { StateProfile } from '../../types/StateProfile.js';
import { ConcretePath } from '@irs/js-factgraph-scala';

const { mockUseFetchStateProfile } = vi.hoisted(() => {
  return {
    mockUseFetchStateProfile: vi.fn(),
  };
});

const { t } = vi.hoisted(() => {
  return {
    t: vi.fn((key: string) => key),
  };
});

vi.mock(`../../hooks/useFetchStateProfile.js`, () => ({
  default: mockUseFetchStateProfile,
}));

vi.mock(`react-i18next`, () => ({
  useTranslation: () => {
    return {
      t: t,
      i18n: {
        language: `en`,
        exists: vi.fn(),
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

describe(`FileYourStateTaxesDetails`, () => {
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
    departmentOfRevenueUrl: `http://www.dor-example-url.example`,
    filingRequirementsUrl: `http://www.filing-requirements-example-url.example`,
  };

  const fetchStateProfileHookResponse: FetchStateProfileHookResponse = {
    stateProfile: stateProfile,
    isFetching: false,
    fetchError: false,
    fetchSuccess: false,
    fetchSkipped: false,
  };

  const fetchStateProfileHookResponseOregon: FetchStateProfileHookResponse = {
    ...fetchStateProfileHookResponse,
    stateProfile: {
      ...stateProfile,
      stateCode: `OR`,
    },
  };

  const fetchStateProfileHookResponseArizona: FetchStateProfileHookResponse = {
    ...fetchStateProfileHookResponse,
    stateProfile: {
      ...stateProfile,
      stateCode: `AZ`,
    },
  };

  const fetchStateProfileHookResponseArizonaWithAcceptedOnlyTrue: FetchStateProfileHookResponse = {
    ...fetchStateProfileHookResponse,
    stateProfile: {
      ...stateProfile,
      stateCode: `AZ`,
      acceptedOnly: true,
    },
  };

  const fetchStateProfileHookResponseCalifornia: FetchStateProfileHookResponse = {
    ...fetchStateProfileHookResponse,
    stateProfile: {
      ...stateProfile,
      stateCode: `CA`,
    },
  };

  beforeEach(() => {
    vi.restoreAllMocks();
    mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponse);
  });

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

  const rejectedStatus: TaxReturnSubmissionStatus = {
    ...pendingStatus,
    status: FEDERAL_RETURN_STATUS.REJECTED,
  };

  const acceptedStatus: TaxReturnSubmissionStatus = {
    ...pendingStatus,
    status: FEDERAL_RETURN_STATUS.ACCEPTED,
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
            <FileYourStateTaxesDetails />
          </SubmissionStatusContext.Provider>
        </TaxReturnsContext.Provider>
      )
    );

    const heading = screen.getByRole(`heading`, { level: 2 });
    const subHeading = screen.getByRole(`heading`, { level: 3 });

    const links = screen.getAllByRole(`link`);
    expect(links.length).toEqual(2);
    const learnAboutStateFilingRequirementsLink = links[0];
    const startStateTaxesLink = links[1];

    const listItems = screen.queryAllByRole(`listitem`);
    return { heading, subHeading, learnAboutStateFilingRequirementsLink, startStateTaxesLink, listItems };
  };

  it(`should render correct content when status is pending`, () => {
    const { heading, subHeading, learnAboutStateFilingRequirementsLink, startStateTaxesLink, listItems } =
      renderComponent();

    expect(heading).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.heading`);
    expect(subHeading).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.subHeading`);
    expect(listItems.length).toEqual(3);

    expect(listItems[0]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepOne`);
    expect(listItems[1]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepTwoDefault`);
    expect(listItems[2]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepThree`);

    expect(learnAboutStateFilingRequirementsLink).toHaveTextContent(
      `taxReturnCard.fileYourStateTaxesDetails.learnAboutStateFilingRequirementsButtonText`
    );
    expect(learnAboutStateFilingRequirementsLink).toHaveProperty(`href`, `${stateProfile.filingRequirementsUrl}/`);
    expect(startStateTaxesLink).toHaveTextContent(
      `taxReturnCard.fileYourStateTaxesDetails.startYourStateTaxesButtonText`
    );
    expect(startStateTaxesLink).toHaveProperty(`href`, `${stateProfile.landingUrl}?ref_location=df_home`);
  });

  it(`should render correct content when status is rejected and user is filing in a 
    state where they can transfer returns`, () => {
    useFact.mockImplementation((path: ConcretePath) => {
      if (path.includes(`/stateCanTransferData`)) {
        return [true];
      }

      return [false];
    });

    const { heading, subHeading, learnAboutStateFilingRequirementsLink, startStateTaxesLink, listItems } =
      renderComponent(rejectedStatus);

    expect(heading).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.heading`);
    expect(subHeading).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.subHeading`);
    expect(listItems.length).toEqual(3);

    expect(listItems[0]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepOneRejectedReturn`);
    expect(listItems[1]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepTwoRejectedReturn`);
    expect(listItems[2]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepThree`);

    expect(learnAboutStateFilingRequirementsLink).toHaveTextContent(
      `taxReturnCard.fileYourStateTaxesDetails.learnAboutStateFilingRequirementsButtonText`
    );
    expect(learnAboutStateFilingRequirementsLink).toHaveProperty(`href`, `${stateProfile.filingRequirementsUrl}/`);
    expect(startStateTaxesLink).toHaveTextContent(
      `taxReturnCard.fileYourStateTaxesDetails.startYourStateTaxesButtonText`
    );
    expect(startStateTaxesLink).toHaveProperty(`href`, `${stateProfile.landingUrl}?ref_location=df_home`);
  });

  it(`should render correct content when state is Oregon`, () => {
    useFact.mockImplementation((path: ConcretePath) => {
      if (path.includes(`/stateCanTransferData`)) {
        return [true];
      }
      return [false];
    });
    mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponseOregon);

    const { heading, subHeading, learnAboutStateFilingRequirementsLink, startStateTaxesLink, listItems } =
      renderComponent(rejectedStatus);

    expect(heading).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.heading`);
    expect(subHeading).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.subHeading`);
    expect(listItems.length).toEqual(3);

    expect(listItems[0]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepOne`);
    expect(listItems[1]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepTwoOregon`);
    expect(listItems[2]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepThree`);

    expect(learnAboutStateFilingRequirementsLink).toHaveTextContent(
      `taxReturnCard.fileYourStateTaxesDetails.learnAboutStateFilingRequirementsButtonText`
    );
    expect(learnAboutStateFilingRequirementsLink).toHaveProperty(`href`, `${stateProfile.filingRequirementsUrl}/`);
    expect(startStateTaxesLink).toHaveTextContent(
      `taxReturnCard.fileYourStateTaxesDetails.startYourStateTaxesButtonText`
    );
    expect(startStateTaxesLink).toHaveProperty(`href`, `${stateProfile.landingUrl}?ref_location=df_home`);
  });

  describe(`Wait Before Return Is Accepted Before Filing State Taxes Message`, () => {
    it(`should render when return is pending in a state that needs to wait before return is accepted`, () => {
      mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponseArizonaWithAcceptedOnlyTrue);
      const { listItems } = renderComponent();

      expect(listItems[1]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepTwoDefault`);
      expect(listItems[1]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.youWillNeedToWait`);
    });

    // eslint-disable-next-line max-len
    it(`should not render when return is pending in a state that does not have to wait before return is accepted`, () => {
      mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponse);
      const { listItems } = renderComponent();

      expect(listItems[1]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepTwoDefault`);
      expect(listItems[1]).not.toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.youWillNeedToWait`);
    });

    // eslint-disable-next-line max-len
    it(`should not render when return is rejected in a state that needs to wait before return is accepted`, () => {
      mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponseArizona);
      useFact.mockImplementation((path: ConcretePath) => {
        if (path.includes(`/stateCanTransferData`)) {
          return [true];
        }
        return [false];
      });

      const { listItems } = renderComponent(rejectedStatus);

      expect(listItems[1]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepTwoRejectedReturn`);
      expect(listItems[1]).not.toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.youWillNeedToWait`);
    });

    it(`should not render when return is accepted in a state that needs to wait before return is accepted`, () => {
      mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponseArizona);
      const { listItems } = renderComponent(acceptedStatus);

      expect(listItems[1]).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stepTwoDefault`);
      expect(listItems[1]).not.toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.youWillNeedToWait`);
    });

    it(`should only render a message telling a user they can file their state taxes after submitting federal return 
      and not render a list of steps if state cannot transfer data and return status is rejected`, () => {
      useFact.mockImplementation((path: ConcretePath) => {
        if (path.includes(`/stateCanTransferData`)) {
          return [false];
        }
        return [false];
      });

      mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponseCalifornia);
      const { listItems } = renderComponent(rejectedStatus);

      expect(listItems.length).toBe(0);
      expect(screen.getByText(/stepOneRejectedReturn/)).toBeInTheDocument();
    });
  });
});
