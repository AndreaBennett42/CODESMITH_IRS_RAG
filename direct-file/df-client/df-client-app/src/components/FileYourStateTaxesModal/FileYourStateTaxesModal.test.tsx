import { render, screen } from '@testing-library/react';
import { wrapComponent } from '../../test/helpers.js';
import FileYourStateTaxesModal, { FileYourStateTaxesModalProps } from './FileYourStateTaxesModal.js';
import { StateProfile } from '../../types/StateProfile.js';
import { vi } from 'vitest';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';
import { TaxReturn, TaxReturnSubmissionStatus } from '../../types/core.js';
import { v4 as uuidv4 } from 'uuid';
import { CURRENT_TAX_YEAR, FEDERAL_RETURN_STATUS } from '../../constants/taxConstants.js';
import { ModalRef } from '@trussworks/react-uswds';
import { RefObject } from 'react';
import { FetchStateProfileHookResponse } from '../../hooks/useFetchStateProfile.js';

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
    useFact: vi.fn(() => [false]),
  };
});

vi.mock(`../../hooks/useFact`, () => ({
  default: useFact,
}));

describe(`FileYourStateTaxesModal`, () => {
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
    departmentOfRevenueUrl: `http://www.dor-example-url.example`,
    filingRequirementsUrl: `http://www.filing-requirements-example-url.example`,
    transferCancelUrl: ``,
    waitingForAcceptanceCancelUrl: ``,
    redirectUrls: [],
    languages: { en: `en`, es: `es` },
    taxSystemName: `FSTSN`,
    acceptedOnly: false,
    customFilingDeadline: null,
  };

  const fetchStateProfileHookResponse: FetchStateProfileHookResponse = {
    stateProfile: stateProfile,
    isFetching: false,
    fetchError: false,
    fetchSuccess: false,
    fetchSkipped: false,
  };

  const mockModalRef: RefObject<ModalRef> = {} as RefObject<ModalRef>;

  const defaultProps: FileYourStateTaxesModalProps = {
    modalRef: mockModalRef,
    stateProfile: stateProfile,
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
            <FileYourStateTaxesModal {...defaultProps} />
          </SubmissionStatusContext.Provider>
        </TaxReturnsContext.Provider>
      )
    );

    const modalHeading = screen.getByRole(`heading`, { level: 2 });
    const modalSubheading = screen.getByRole(`heading`, { level: 3 });
    const modalLinks = screen.getAllByRole(`link`);

    const learnMoreAboutStateFilingRequrementsHeading = modalLinks[0];
    const stateDorLink = modalLinks[1];

    const modalBody = screen.getByText(/taxReturnCard.fileYourStateTaxesDetails.details/);

    return { modalHeading, modalSubheading, learnMoreAboutStateFilingRequrementsHeading, stateDorLink, modalBody };
  };

  it(`should render correct text content`, () => {
    mockUseFetchStateProfile.mockReturnValue(fetchStateProfileHookResponse);

    const { modalHeading, modalSubheading, learnMoreAboutStateFilingRequrementsHeading, stateDorLink, modalBody } =
      renderComponent();
    expect(modalHeading).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.modalHeading`);
    expect(modalSubheading).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.subheading`);

    expect(learnMoreAboutStateFilingRequrementsHeading).toHaveTextContent(
      `taxReturnCard.fileYourStateTaxesDetails.learnAboutStateFilingRequirementsButtonText`
    );
    expect(learnMoreAboutStateFilingRequrementsHeading).toHaveProperty(
      `href`,
      `${stateProfile.filingRequirementsUrl}/`
    );

    expect(stateDorLink).toHaveTextContent(`taxReturnCard.fileYourStateTaxesDetails.stateDorSite`);
    expect(stateDorLink).toHaveProperty(`href`, `${stateProfile.departmentOfRevenueUrl}/`);

    expect(modalBody).toBeInTheDocument();
  });
});
