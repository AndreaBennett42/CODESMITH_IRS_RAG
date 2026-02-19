import { DollarFactory } from '@irs/js-factgraph-scala';
import flowNodes from '../../flow/flow.js';
import { createFlowConfig } from '../../flow/flowConfig.js';
import {
  createBooleanWrapper,
  createEnumWrapper,
  createStringWrapper,
  createTinWrapper,
} from '../persistenceWrappers.js';
import { FlowSetFactInput, checkFlowSetsFacts } from './checkFlowSetsFacts.js';

export const uuid = `959c03d1-af4a-447f-96aa-d19397048a44`;
const flow = createFlowConfig(flowNodes);

describe(`Interest loop always sets interest facts`, () => {
  const input: Omit<FlowSetFactInput, 'leaveOptionalFieldsBlank'> = {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    startingScreenRoute: flow.collectionLoopsByName.get(`/interestReports`)!.screens[0].screenRoute,
    startingFactState: {
      '/interestReports': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [uuid] } },
      '/form1099Gs': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [] } },
      '/socialSecurityReports': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [] } },
      [`/hasForeignAccounts`]: createBooleanWrapper(false),
      [`/isForeignTrustsGrantor`]: createBooleanWrapper(false),
      [`/hasForeignTrustsTransactions`]: createBooleanWrapper(false),
    },
    expectedCompleteFacts: [`/form1099Withholding`, `/hasInterestReports`, `/taxExemptInterest`, `/interestIncome`],
    expectedStartingFactState: {},
    collectionId: uuid,
    factValuesToSet: {
      // The $2000 value ensures that we get knocked out in some paths.
      '/interestReports/*/no1099Amount': [DollarFactory(`100`).right, DollarFactory(`2000`).right],
      '/interestReports/*/1099Amount': [DollarFactory(`100`).right, DollarFactory(`2000`).right],
    },
  };
  it(`All complete interest facts get set as complete`, () => {
    checkFlowSetsFacts({ ...input, leaveOptionalFieldsBlank: false });
  });
  it(`All complete interest facts get set as complete when leaving optional fields blank`, () => {
    checkFlowSetsFacts({ ...input, leaveOptionalFieldsBlank: true });
  });
});

/**
 * This test is skipped because it takes too long to run.
 * This is a real bummer.
 **/
describe.skip(`Jobs loop always sets employment income facts`, () => {
  const input: Omit<FlowSetFactInput, 'leaveOptionalFieldsBlank'> = {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    startingScreenRoute: flow.subcategoriesByRoute.get(`/flow/income/jobs`)!.screens[0].screenRoute,
    startingFactState: {
      '/formW2s': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [uuid] } },
      '/filers': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [uuid] } },
      '/address': {
        $type: `gov.irs.factgraph.persisters.AddressWrapper`,
        item: { streetAddress: `111 Addy`, city: `Washington`, postalCode: `20001`, stateOrProvence: `DC` },
      },
      [`/filers/#${uuid}/isPrimaryFiler`]: createBooleanWrapper(true),
      [`/filers/#${uuid}/firstName`]: createStringWrapper(`Test`),
      [`/filers/#${uuid}/writableMiddleInitial`]: createStringWrapper(`Test`),
      [`/filers/#${uuid}/lastName`]: createStringWrapper(`Test`),
      [`/filers/#${uuid}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
    },
    expectedCompleteFacts: [
      `/wages`,
      `/formW2Withholding`,
      `/socialSecurityTaxesWithheld`,
      `/hasW2s`,
      `/formW2s/*/isComplete`,
      `/formW2s/*/filer`,
      `/formW2s/*/standardOrNonStandardCd`,
      `/formW2s/*/usedTin`,
      `/formW2s/*/ein`,
      `/formW2s/*/employerName`,
      `/formW2s/*/employerAddress`,
      `/formW2s/*/wages`,
      `/formW2s/*/federalWithholding`,
      `/formW2s/*/oasdiWages`,
      `/formW2s/*/oasdiWithholding`,
      `/formW2s/*/medicareWages`,
      `/formW2s/*/medicareWithholding`,
      `/formW2s/*/oasdiTips`,
      `/formW2s/*/allocatedTips`,
      `/formW2s/*/nonQualifiedPlans`,
      `/formW2s/*/address`,
    ],
    expectedStartingFactState: {},
    collectionId: uuid,
    factValuesToSet: {
      '/formW2s/*/writableAllocatedTips': [DollarFactory(`0`).right, DollarFactory(`1000`).right],
      '/formW2s/*/writableDependentCareBenefits': [DollarFactory(`0`).right, DollarFactory(`1000`).right],
      '/formW2s/*/writableNonQualifiedPlans': [DollarFactory(`0`).right, DollarFactory(`1000`).right],
    },
  };
  it(`All complete w2 facts get set as complete`, () => {
    checkFlowSetsFacts({ ...input, leaveOptionalFieldsBlank: false });
  });
  it(`All complete w2 facts get set as complete when filer's tin is not an ssn`, () => {
    checkFlowSetsFacts({
      ...input,
      startingFactState: {
        ...input.startingFactState,
        [`/filers/#${uuid}/tin`]: createTinWrapper({ area: `100`, group: `55`, serial: `5555` }),
      },
      leaveOptionalFieldsBlank: false,
    });
  });
  it(`All complete w2 facts get set as complete when leaving optional fields blank`, () => {
    checkFlowSetsFacts({ ...input, leaveOptionalFieldsBlank: true });
  });
});

describe(`Unemployment loop always sets unemployment income facts`, () => {
  const input: Omit<FlowSetFactInput, 'leaveOptionalFieldsBlank'> = {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    startingScreenRoute: flow.subcategoriesByRoute.get(`/flow/income/unemployment`)!.screens[0].screenRoute,
    startingFactState: {
      '/form1099Gs': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [uuid] } },
      '/interestReports': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [] } },
      '/socialSecurityReports': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [] } },
      '/filers': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [uuid] } },
      [`/filers/#${uuid}/isPrimaryFiler`]: createBooleanWrapper(true),
      [`/filers/#${uuid}/firstName`]: createStringWrapper(`Test`),
      [`/filers/#${uuid}/writableMiddleInitial`]: createStringWrapper(`Test`),
      [`/filers/#${uuid}/lastName`]: createStringWrapper(`Test`),
      [`/filers/#${uuid}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
    },
    expectedStartingFactState: {},
    expectedCompleteFacts: [`/form1099Withholding`, `/unemploymentCompensation`, `/otherIncome`],
    collectionId: uuid,
    factValuesToSet: {},
  };
  it(`All complete unemployment facts get set as complete`, () => {
    checkFlowSetsFacts({ ...input, leaveOptionalFieldsBlank: false });
  });
  it(`All complete unemployment facts get set as complete when leaving optional fields blank`, () => {
    checkFlowSetsFacts({ ...input, leaveOptionalFieldsBlank: true });
  });
});

describe(`Social security loop always sets social security income facts`, () => {
  const input: Omit<FlowSetFactInput, 'leaveOptionalFieldsBlank'> = {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    startingScreenRoute: flow.subcategoriesByRoute.get(`/flow/income/social-security`)!.screens[0].screenRoute,
    startingFactState: {
      '/form1099Gs': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [] } },
      '/interestReports': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [] } },
      '/socialSecurityReports': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [uuid] } },
      '/filers': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [uuid] } },
      [`/filers/#${uuid}/isPrimaryFiler`]: createBooleanWrapper(true),
      [`/filers/#${uuid}/firstName`]: createStringWrapper(`Test`),
      [`/filers/#${uuid}/writableMiddleInitial`]: createStringWrapper(`Test`),
      [`/filers/#${uuid}/lastName`]: createStringWrapper(`Test`),
      [`/filers/#${uuid}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
    },
    expectedStartingFactState: {},
    expectedCompleteFacts: [`/form1099Withholding`, `/socialSecurityBenefits`],
    collectionId: uuid,
    factValuesToSet: {},
  };
  it(`All social security facts facts get set as complete`, () => {
    checkFlowSetsFacts({ ...input, leaveOptionalFieldsBlank: false });
  });
  it(`All social security facts facts get set as complete when leaving optional fields blank`, () => {
    checkFlowSetsFacts({ ...input, leaveOptionalFieldsBlank: true });
  });
});

describe(`Alaska Permanent Fund (PFD) loop always sets PFD income facts`, () => {
  const input: Omit<FlowSetFactInput, 'leaveOptionalFieldsBlank'> = {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    startingScreenRoute: flow.subcategoriesByRoute.get(`/flow/income/alaska-pfd`)!.screens[0].screenRoute,
    startingFactState: {
      '/form1099Gs': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [] } },
      '/interestReports': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [] } },
      '/socialSecurityReports': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [] } },
      '/form1099Miscs': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [uuid] } },
      '/filers': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [uuid] } },
      [`/filers/#${uuid}/isPrimaryFiler`]: createBooleanWrapper(true),
      [`/filers/#${uuid}/firstName`]: createStringWrapper(`Test`),
      [`/filers/#${uuid}/writableMiddleInitial`]: createStringWrapper(`Test`),
      [`/filers/#${uuid}/lastName`]: createStringWrapper(`Test`),
      [`/filers/#${uuid}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`AK`, `/filerResidenceAndIncomeState`),
    },
    expectedStartingFactState: {},
    expectedCompleteFacts: [
      `/isApfSectionComplete`,
      `/otherIncome`,
      `/form1099Withholding`,
      `/alaskaPfd1099s/*/writableOtherIncome`,
      `/alaskaPfd1099s/*/federalWithholding`,
      `/alaskaPfd1099s/*/federalIncomeTaxWithheldRounded`,
      `/alaskaPfd1099s/*/payer`,
      `/alaskaPfd1099s/*/payer/tin`,
      `/alaskaPfd1099s/*/isComplete`,
      `/alaskaPfdIncome`,
    ],
    collectionId: uuid,
    factValuesToSet: {},
  };
  it(`All APF facts facts get set as complete`, () => {
    checkFlowSetsFacts({ ...input, leaveOptionalFieldsBlank: false });
  });
  it(`All APF facts facts get set as complete when leaving optional fields blank`, () => {
    checkFlowSetsFacts({ ...input, leaveOptionalFieldsBlank: true });
  });
});
