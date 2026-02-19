// API response mock data
import { FactValue } from '../../types/core.js';
import { createBooleanWrapper, createStringWrapper } from '../persistenceWrappers.js';

const stateOrProvince = `DC`;
export const facts: { [path: string]: FactValue } = {
  '/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/canBeClaimed': createBooleanWrapper(false),
  '/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/occupation': createStringWrapper(`mat`),
  '/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/middleInitial': createStringWrapper(`R`),
  '/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/employerName': createStringWrapper(`the USG`),
  '/formW2s': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: {
      items: [`3d1946aa-7280-43d4-b5c9-5fde6a6ba28c`],
    },
  },
  '/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/medicareWithholding': {
    $type: `gov.irs.factgraph.persisters.DollarWrapper`,
    item: `5000.00`,
  },
  '/taxYear': {
    $type: `gov.irs.factgraph.persisters.IntWrapper`,
    item: 2022,
  },
  '/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/oasdiTips': {
    $type: `gov.irs.factgraph.persisters.DollarWrapper`,
    item: `0.00`,
  },
  '/familyAndHousehold': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: {
      items: [],
    },
  },
  '/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/lastName': createStringWrapper(`Lake`),
  '/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/oasdiWages': {
    $type: `gov.irs.factgraph.persisters.DollarWrapper`,
    item: `90000.00`,
  },
  '/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/combatPay': {
    $type: `gov.irs.factgraph.persisters.DollarWrapper`,
    item: `0.00`,
  },
  '/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/isPrimaryFiler': createBooleanWrapper(true),
  '/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/oasdiWithholding': {
    $type: `gov.irs.factgraph.persisters.DollarWrapper`,
    item: `5000.00`,
  },
  '/maritalStatus': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`single`],
      options: [`married`, `divorced`, `single`, `widowed`],
      enumId: `maritalStatus`,
    },
  },
  '/receivedDigitalAssets': createBooleanWrapper(false),
  '/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/tin': createStringWrapper(`121123121`),
  '/wantsStandardDeduction': createBooleanWrapper(true),
  '/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/medicareWages': {
    $type: `gov.irs.factgraph.persisters.DollarWrapper`,
    item: `90000.00`,
  },
  '/phone': {
    $type: `gov.irs.factgraph.persisters.E164Wrapper`,
    item: {
      $type: `gov.irs.factgraph.types.UsPhoneNumber`,
      areaCode: `202`,
      officeCode: `555`,
      lineNumber: `0100`,
    },
  },
  '/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/isBlind': createBooleanWrapper(false),
  '/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/federalWithholding': {
    $type: `gov.irs.factgraph.persisters.DollarWrapper`,
    item: `30000.00`,
  },
  '/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/firstName': createStringWrapper(`Todd`),
  '/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/wages': {
    $type: `gov.irs.factgraph.persisters.DollarWrapper`,
    item: `80000.00`,
  },
  '/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/ein': createStringWrapper(`111121121`),
  '/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/dateOfBirth': {
    $type: `gov.irs.factgraph.persisters.DayWrapper`,
    item: {
      date: `2000-01-01`,
    },
  },
  '/address': {
    $type: `gov.irs.factgraph.persisters.AddressWrapper`,
    item: {
      streetAddress: `736 Jackson Place NW`,
      city: `Washington`,
      postalCode: `20503`,
      stateOrProvence: stateOrProvince,
    },
  },
  '/filingStatus': {
    $type: `gov.irs.factgraph.persisters.EnumWrapper`,
    item: {
      value: [`single`],
      options: [
        `single`,
        `qualifiedSurvivingSpouse`,
        `headOfHousehold`,
        `marriedFilingSeparately`,
        `marriedFilingJointly`,
      ],
      enumId: `filingStatus`,
    },
  },
  '/filers': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: {
      items: [`6b1259fd-8cdb-4efe-bcc8-ad40e604c98b`],
    },
  },
  '/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/filer': {
    $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
    item: {
      id: `6b1259fd-8cdb-4efe-bcc8-ad40e604c98b`,
    },
  },
};

export const taxreturnsError = {
  message: `error msg`,
  errors: {
    additionalProp1: `prop1 is wrong`,
  },
};
