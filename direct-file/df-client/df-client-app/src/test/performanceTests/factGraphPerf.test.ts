/* eslint-disable no-console */
import { describe, it } from 'vitest';
import { ConcretePath } from '@irs/js-factgraph-scala';
import { setupFactGraph } from '../setupFactGraph.js';
import { createBooleanWrapper } from '../persistenceWrappers.js';

const returnFacts = {
  [`/hasCdccCarryoverAmountFromPriorTaxYear`]: createBooleanWrapper(false),
  [`/flowKnockoutHouseholdEmployee`]: createBooleanWrapper(false),
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/tin': {
    $type: `gov.irs.factgraph.persisters.TinWrapper`,
    item: {
      area: `453`,
      group: `45`,
      serial: `4432`,
    },
  },
  '/payViaAch': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/filedLastYear': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/primaryFilerW2And1099IntInScopedState': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`noForms`],
      enumOptionsPath: `/primaryFilerW2And1099IntStateOptions`,
    },
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/tin': {
    $type: `gov.irs.factgraph.persisters.TinWrapper`,
    item: {
      area: `564`,
      group: `85`,
      serial: `6464`,
    },
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/dateOfBirth': {
    $type: `gov.irs.factgraph.persisters.DayWrapper`,
    item: {
      date: `1991-12-14`,
    },
  },
  '/filers/#04fbb751-3fcb-48a6-8280-baed443106ef/hasIpPin': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/retirementPlan': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/filerResidenceAndIncomeState': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`ak`],
      enumOptionsPath: `/scopedStateOptions`,
    },
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/married': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/wantsCustomLanguage': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/employerAddress': {
    $type: `gov.irs.factgraph.persisters.AddressWrapper`,
    item: {
      streetAddress: `234 Main St`,
      city: `Washington`,
      postalCode: `23434`,
      stateOrProvence: `AL`,
      country: ``,
    },
  },
  '/incomeSourcesSupported': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/wasK12Educators': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`neither`],
      enumOptionsPath: `/k12EducatorOptions`,
    },
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/nonstandardOrCorrectedChoice': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`nonstandard`],
      enumOptionsPath: `/w2NonstandardCorrectedOptions`,
    },
  },
  '/filers/#04fbb751-3fcb-48a6-8280-baed443106ef/isPrimaryFiler': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/retirementPlan': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/employerName': {
    $type: `gov.irs.factgraph.persisters.StringWrapper`,
    item: `Burger Company`,
  },
  '/maritalStatus': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`single`],
      enumOptionsPath: `/maritalStatusOptions`,
    },
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/filer': {
    $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
    item: {
      id: `04fbb751-3fcb-48a6-8280-baed443106ef`,
    },
  },
  '/interestReports': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: {
      items: [],
    },
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/writableWages': {
    $type: `gov.irs.factgraph.persisters.DollarWrapper`,
    item: `70234.00`,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/unableToCareForSelf': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/permanentTotalDisability': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/grossIncomeTest': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/flowHasSeenAmount': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/wantsThirdPartyDesignee': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/statutoryEmployee': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/socialSecurityReports': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: {
      items: [],
    },
  },
  '/formW2s': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: {
      items: [`b8eb908e-8eb5-4d9e-beaa-ab023680267f`, `0c76d365-b8c7-4cf7-967b-f25fac66df14`],
    },
  },
  '/receivedImproperClaims': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/tpClaims': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/presidentalCampaignDesignation': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`filer`],
      enumOptionsPath: `/presidentalCampaignOptions`,
    },
  },
  '/filers/#04fbb751-3fcb-48a6-8280-baed443106ef/isUsCitizenFullYear': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/lastName': {
    $type: `gov.irs.factgraph.persisters.StringWrapper`,
    item: `Smith`,
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/hasRRTACodes': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/flowHasSeenDeductions': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/socialSecurityReportsIsDone': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/fullTimeStudent': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/phone': {
    $type: `gov.irs.factgraph.persisters.E164Wrapper`,
    item: {
      $type: `gov.irs.factgraph.types.UsPhoneNumber`,
      areaCode: `729`,
      officeCode: `320`,
      lineNumber: `3205`,
    },
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/tinType': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`ssn`],
      enumOptionsPath: `/tinTypeOptions`,
    },
  },
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/addressMatchesReturn': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/hasRRTACodes': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/statutoryEmployee': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/disposedDigitalAssets': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/hasSeenLastAvailableScreen': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/nonstandardOrCorrectedChoice': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`corrected`],
      enumOptionsPath: `/w2NonstandardCorrectedOptions`,
    },
  },
  '/filers/#6eddde04-d4c7-43bc-b78d-08a764eee0ec/isPrimaryFiler': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/employerAddress': {
    $type: `gov.irs.factgraph.persisters.AddressWrapper`,
    item: {
      streetAddress: `23423 South St`,
      city: `Wimpleton`,
      postalCode: `23434`,
      stateOrProvence: `AL`,
      country: ``,
    },
  },
  '/flowHasSeenCreditsIntroNoCredits': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/ein': {
    $type: `gov.irs.factgraph.persisters.EinWrapper`,
    item: {
      prefix: `45`,
      serial: `5454564`,
    },
  },
  '/filers/#04fbb751-3fcb-48a6-8280-baed443106ef/flowIpPinReady': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/addressMatchesReturn': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/filers/#04fbb751-3fcb-48a6-8280-baed443106ef/isBlind': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/deceased': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/receivedDigitalAssets': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/familyAndHousehold': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: {
      items: [`8a257480-cb94-451a-9196-c08b7b116867`],
    },
  },
  '/hasSeenReviewScreen': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/form1099Gs': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: {
      items: [],
    },
  },
  '/filers/#04fbb751-3fcb-48a6-8280-baed443106ef/firstName': {
    $type: `gov.irs.factgraph.persisters.StringWrapper`,
    item: `Homer`,
  },
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/hasSeenLastAvailableScreen': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/thirdPartySickPay': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/interestReportsIsDone': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/filers/#04fbb751-3fcb-48a6-8280-baed443106ef/dateOfBirth': {
    $type: `gov.irs.factgraph.persisters.DayWrapper`,
    item: {
      date: `1950-12-12`,
    },
  },
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/employerName': {
    $type: `gov.irs.factgraph.persisters.StringWrapper`,
    item: `Candy Store`,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/childRelationship': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`biologicalChild`],
      enumOptionsPath: `/relationshipOptions`,
    },
  },
  '/filers/#04fbb751-3fcb-48a6-8280-baed443106ef/occupation': {
    $type: `gov.irs.factgraph.persisters.StringWrapper`,
    item: `Worker bee`,
  },
  '/email': {
    $type: `gov.irs.factgraph.persisters.EmailAddressWrapper`,
    item: {
      email: `user.0000@example.com`,
    },
  },
  '/filers/#04fbb751-3fcb-48a6-8280-baed443106ef/writableMiddleInitial': {
    $type: `gov.irs.factgraph.persisters.StringWrapper`,
    item: `C`,
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/writableStateWages': {
    $type: `gov.irs.factgraph.persisters.DollarWrapper`,
    item: `5444.00`,
  },
  '/formW2sIsDone': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/ein': {
    $type: `gov.irs.factgraph.persisters.EinWrapper`,
    item: {
      prefix: `58`,
      serial: `5654585`,
    },
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/tin': {
    $type: `gov.irs.factgraph.persisters.TinWrapper`,
    item: {
      area: `453`,
      group: `45`,
      serial: `4432`,
    },
  },
  '/filers/#04fbb751-3fcb-48a6-8280-baed443106ef/canBeClaimed': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/hasIpPin': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/address': {
    $type: `gov.irs.factgraph.persisters.AddressWrapper`,
    item: {
      streetAddress: `123 Main St`,
      city: `Washington`,
      postalCode: `20001`,
      stateOrProvence: `DC`,
      streetAddressLine2: `Apt 2`,
      country: ``,
    },
  },
  '/filingStatus': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`single`],
      enumOptionsPath: `/filingStatusOptions`,
    },
  },
  '/filers': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: {
      items: [`04fbb751-3fcb-48a6-8280-baed443106ef`, `6eddde04-d4c7-43bc-b78d-08a764eee0ec`],
    },
  },
  '/filers/#04fbb751-3fcb-48a6-8280-baed443106ef/lastName': {
    $type: `gov.irs.factgraph.persisters.StringWrapper`,
    item: `Simpson`,
  },
  '/familyAndHouseholdIsDone': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/form1099GsIsDone': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/languagePreference': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`vietnamese`],
      enumOptionsPath: `/languageOptions`,
    },
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/residencyDuration': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`allYear`],
      enumOptionsPath: `/residencyDurationOptions`,
    },
  },
  '/wantsCommsFormat': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/filer': {
    $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
    item: {
      id: `04fbb751-3fcb-48a6-8280-baed443106ef`,
    },
  },
  '/filers/#04fbb751-3fcb-48a6-8280-baed443106ef/tin': {
    $type: `gov.irs.factgraph.persisters.TinWrapper`,
    item: {
      area: `453`,
      group: `45`,
      serial: `4432`,
    },
  },
  '/selfSelectPin': {
    $type: `gov.irs.factgraph.persisters.PinWrapper`,
    item: {
      pin: `12345`,
    },
  },
  '/formW2s/#b8eb908e-8eb5-4d9e-beaa-ab023680267f/thirdPartySickPay': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/formW2s/#0c76d365-b8c7-4cf7-967b-f25fac66df14/writableWages': {
    $type: `gov.irs.factgraph.persisters.DollarWrapper`,
    item: `50000.00`,
  },
  '/hadStudentLoanInterestPayments': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/isUsCitizenFullYear': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/writableQrSupportTest': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/paidEstimatedTaxesOrFromLastYear': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/tpPaidMostOfHomeUpkeep': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
  '/familyAndHousehold/#8a257480-cb94-451a-9196-c08b7b116867/firstName': {
    $type: `gov.irs.factgraph.persisters.StringWrapper`,
    item: `John`,
  },
  '/hasForeignAccounts': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/isForeignTrustsGrantor': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/hasForeignTrustsTransactions': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: false,
  },
  '/cdccCareProviders': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: {
      items: [],
    },
  },
  '/form1099Rs': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: {
      items: [],
    },
  },
};

// This does not belong in a vitest test. It is not a test.
// But I was struggling to run a node script in `df-client-app`, and this lets us run and console log things.
describe(`Fact Graph performance`, () => {
  it(`On a relatively big return`, () => {
    const { factGraph } = setupFactGraph(returnFacts);
    let startTime = performance.now();
    factGraph.get(`/owesBalance` as ConcretePath);
    console.log(`/owesBalance took ${performance.now() - startTime}ms on first fetch`);

    startTime = performance.now();
    factGraph.get(`/owesBalance` as ConcretePath);
    console.log(`/owesBalance took ${performance.now() - startTime}ms on second fetch`);

    startTime = performance.now();
    factGraph.get(`/balanceDue` as ConcretePath);
    console.log(`/balanceDue took ${performance.now() - startTime}ms on first fetch`);

    startTime = performance.now();
    factGraph.get(`/balanceDue` as ConcretePath);
    console.log(`/balanceDue took ${performance.now() - startTime}ms on second fetch`);
    startTime = performance.now();
    factGraph.get(`/balanceDue` as ConcretePath);
    console.log(`/balanceDue took ${performance.now() - startTime}ms on third fetch`);
    startTime = performance.now();
    const bd = factGraph.get(`/balanceDue` as ConcretePath);
    console.log(`/balanceDue took ${performance.now() - startTime}ms on fourth fetch`);
    expect(bd.complete).toBe(true);
  });
});
