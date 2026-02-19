import { ConcretePath, DayFactory } from '@irs/js-factgraph-scala';
import { describe, it, expect } from 'vitest';
import {
  createCollectionWrapper,
  createBooleanWrapper,
  createDayWrapper,
  createStringWrapper,
  createTinWrapper,
  createIpPinWrapper,
  createEnumWrapper,
  createAddressWrapper,
  createPhoneWrapper,
} from '../persistenceWrappers.js';
import { Path } from '../../flow/Path.js';
import { FactValue } from '../../types/core.js';
import { CURRENT_TAX_YEAR } from '../../constants/taxConstants.js';
import { setupFactGraph } from '../setupFactGraph.js';

const uuid = `959c03d1-af4a-447f-96aa-d19397048a44`;
const spouseId = `859c03d1-af4a-447f-96aa-d19397048a48`;
const primaryFilerId = uuid;

const baseFilersData = {
  '/filers': createCollectionWrapper([primaryFilerId, spouseId]),
  [`/filers/#${primaryFilerId}/isPrimaryFiler`]: createBooleanWrapper(true),
  [`/filers/#${spouseId}/isPrimaryFiler`]: createBooleanWrapper(false),
  [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
  [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`1987-01-02`),
  // This TIN is a valid SSN
  [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
};

describe(`Basic filer info`, () => {
  it(`A fullName property is derived from first and last name`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/fullName`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/firstName`]: createStringWrapper(`Test`),
      [`/filers/#${primaryFilerId}/lastName`]: createStringWrapper(`Testerson`),
    });

    expect(factGraph.get(Path.concretePath(`/filers/*/fullName`, primaryFilerId)).get).toBe(`Test Testerson`);
  });

  it(`Address will parse into a single line`, () => {
    const { factGraph } = setupFactGraph({
      '/address': {
        $type: `gov.irs.factgraph.persisters.AddressWrapper`,
        item: { streetAddress: `111 Addy`, city: `Washington`, postalCode: `20001`, stateOrProvence: `DC` },
      },
    });
    expect(factGraph.get(`/streetAddressAs1Line` as ConcretePath).get).toBe(`111 Addy`);

    const { factGraph: factGraph2 } = setupFactGraph({
      '/address': {
        $type: `gov.irs.factgraph.persisters.AddressWrapper`,
        item: {
          streetAddress: `111 Addy`,
          city: `Washington`,
          postalCode: `20001`,
          stateOrProvence: `DC`,
          streetAddressLine2: `Apt 3`,
        },
      },
    });
    expect(factGraph2.get(`/streetAddressAs1Line` as ConcretePath).get).toBe(`111 Addy Apt 3`);
  });

  it(`A fullName property is derived from first name, middle initial, and last name`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/fullName`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/firstName`]: createStringWrapper(`Test`),
      [`/filers/#${primaryFilerId}/writableMiddleInitial`]: createStringWrapper(`E`),
      [`/filers/#${primaryFilerId}/lastName`]: createStringWrapper(`Testerson`),
    });

    expect(factGraph.get(Path.concretePath(`/filers/*/fullName`, primaryFilerId)).get).toBe(`Test E Testerson`);
  });

  it(`/primaryFiler and /secondaryFiler are aliases to those filers' info`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFiler`, `/secondaryFiler`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/firstName`]: createStringWrapper(`Test`),
      [`/filers/#${primaryFilerId}/lastName`]: createStringWrapper(`Testerson`),
      [`/filers/#${spouseId}/firstName`]: createStringWrapper(`Spousey`),
      [`/filers/#${spouseId}/lastName`]: createStringWrapper(`Testerson`),
    });

    expect(factGraph.get(Path.concretePath(`/primaryFiler/fullName`, null)).get).toBe(`Test Testerson`);
    expect(factGraph.get(Path.concretePath(`/secondaryFiler/fullName`, null)).get).toBe(`Spousey Testerson`);
  });

  it(`A filer's spouse is a secondary filer`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/isSecondaryFiler`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
    });

    expect(factGraph.get(Path.concretePath(`/primaryFiler/isSecondaryFiler`, null)).get).toBe(false);
    expect(factGraph.get(Path.concretePath(`/secondaryFiler/isSecondaryFiler`, null)).get).toBe(true);
  });

  it(`A filer with income in an unsupported state will be KO'd`, ({ task }) => {
    task.meta.testedFactPaths = [`/flowKnockoutUnsupportedState`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`noneOfThese`, `/scopedStateOptions`),
      [`/filingStatus`]: createEnumWrapper(`single`, `/filingStatusOptions`),
    });

    expect(factGraph.get(Path.concretePath(`/flowKnockoutUnsupportedState`, null)).get).toBe(true);
  });

  it(`A return whose primary filer who provides an IP Pin will have that fact recorded`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFilerHasIpPin`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/hasIpPin`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/identityPin`]: createIpPinWrapper(`123456`),
    });

    expect(factGraph.get(Path.concretePath(`/primaryFilerHasIpPin`, null)).get).toBe(true);
  });

  it(`A return whose primary filer doesn't have an IP Pin will have that fact recorded`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFilerHasIpPin`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/hasIpPin`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(Path.concretePath(`/primaryFilerHasIpPin`, null)).get).toBe(false);
  });

  it(`A return whose primary filer has an IP PIN but hasn't
      completed the PIN entry flow should be flagged as missing the PIN`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFilerIsMissingIpPin`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/hasIpPin`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/flowIpPinReady`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(Path.concretePath(`/primaryFilerIsMissingIpPin`, null)).get).toBe(true);
  });

  it(`A return whose primary filer has an IP PIN and has
      completed the PIN entry flow shouldn't be flagged as missing the PIN`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFilerIsMissingIpPin`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/hasIpPin`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/flowIpPinReady`]: createBooleanWrapper(true),
    });

    expect(factGraph.get(Path.concretePath(`/primaryFilerIsMissingIpPin`, null)).get).toBe(false);
  });

  it(`A return whose primary filer doesn't have an IP PIN shouldn't be flagged as missing the PIN`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFilerIsMissingIpPin`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/hasIpPin`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(Path.concretePath(`/primaryFilerIsMissingIpPin`, null)).get).toBe(false);
  });

  it(`A return in which either the primary filer is younger than 16 or the filing status
      is MFJ and the secondary filer is younger than 16 should be KO'd`, ({ task }) => {
    task.meta.testedFactPaths = [`/flowKnockoutFilerAgeYoungerThan16`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 15}-06-01`),
    });

    expect(factGraph.get(Path.concretePath(`/flowKnockoutFilerAgeYoungerThan16`, null)).get).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 15}-06-01`),
    });

    expect(factGraph2.get(Path.concretePath(`/flowKnockoutFilerAgeYoungerThan16`, null)).get).toBe(true);
  });

  it(`A return in which either the primary filer is 16 or older or the filing status
      is MFJ and both filers are 16 or older should not be KO'd`, ({ task }) => {
    task.meta.testedFactPaths = [`/flowKnockoutFilerAgeYoungerThan16`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 16}-06-01`),
    });

    expect(factGraph.get(Path.concretePath(`/flowKnockoutFilerAgeYoungerThan16`, null)).get).toBe(false);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 16}-06-01`),
      [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 16}-06-01`),
    });

    expect(factGraph2.get(Path.concretePath(`/flowKnockoutFilerAgeYoungerThan16`, null)).get).toBe(false);
  });

  it(`if the primary MFJ filer is a citizen, eitherFilerUSCitizenOrNational should indicate that`, ({ task }) => {
    task.meta.testedFactPaths = [`/eitherFilerUSCitizenOrNational`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    });

    expect(factGraph.get(Path.concretePath(`/eitherFilerUSCitizenOrNational`, null)).get).toBe(true);
  });

  it(`if the secondary MFJ filer is a citizen, eitherFilerUSCitizenOrNational should indicate that`, ({ task }) => {
    task.meta.testedFactPaths = [`/eitherFilerUSCitizenOrNational`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    });

    expect(factGraph.get(Path.concretePath(`/eitherFilerUSCitizenOrNational`, null)).get).toBe(true);
  });

  it(`if the primary MFJ filer is a US national, eitherFilerUSCitizenOrNational should indicate that`, ({ task }) => {
    task.meta.testedFactPaths = [`/eitherFilerUSCitizenOrNational`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${primaryFilerId}/writableIsNational`]: createBooleanWrapper(true),
    });

    expect(factGraph.get(Path.concretePath(`/eitherFilerUSCitizenOrNational`, null)).get).toBe(true);
  });

  it(`if the secondary MFJ filer is a US national, eitherFilerUSCitizenOrNational should indicate that`, ({ task }) => {
    task.meta.testedFactPaths = [`/eitherFilerUSCitizenOrNational`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${spouseId}/writableIsNational`]: createBooleanWrapper(true),
    });

    expect(factGraph.get(Path.concretePath(`/eitherFilerUSCitizenOrNational`, null)).get).toBe(true);
  });

  it(`if neither the primary nor secondary MFJ filer is a US citizen or national,
      eitherFilerUSCitizenOrNational should indicate that `, ({ task }) => {
    task.meta.testedFactPaths = [`/eitherFilerUSCitizenOrNational`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableIsNational`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableIsNational`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(Path.concretePath(`/eitherFilerUSCitizenOrNational`, null)).get).toBe(false);
  });

  it(`If either MFJ filer can be claimed as a dependent, filersCouldntBeDependents should be false to indicate that`, ({
    task,
  }) => {
    task.meta.testedFactPaths = [`/filersCouldntBeDependents`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/MFJRequiredToFile`]: createBooleanWrapper(false),
      [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
    });
    expect(factGraph.get(Path.concretePath(`/filersCouldntBeDependents`, null)).get).toBe(false);
  });

  it(`If a filer with any status besides MFJ is a dependent,
      filersCouldntBeDependents should be false to indicate that`, ({ task }) => {
    task.meta.testedFactPaths = [`/filersCouldntBeDependents`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
      [`/filingStatus`]: createEnumWrapper(`single`, `/filingStatusOptions`),
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
    });
    expect(factGraph.get(Path.concretePath(`/filersCouldntBeDependents`, null)).get).toBe(false);
  });

  it(`If neither MFJ filer is not a dependent, filersCouldntBeDependents should be true to indicate that`, ({
    task,
  }) => {
    task.meta.testedFactPaths = [`/filersCouldntBeDependents`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
      [`/MFJRequiredToFile`]: createBooleanWrapper(true),
      [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
    });
    expect(factGraph.get(Path.concretePath(`/filersCouldntBeDependents`, null)).get).toBe(true);
  });

  it(`If a filer with any status besides MFJ is not a dependent,
      filersCouldntBeDependents should be true to indicate that`, ({ task }) => {
    task.meta.testedFactPaths = [`/filersCouldntBeDependents`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
      [`/filingStatus`]: createEnumWrapper(`single`, `/filingStatusOptions`),
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
    });
    expect(factGraph.get(Path.concretePath(`/filersCouldntBeDependents`, null)).get).toBe(true);
  });

  describe(`Filer expects to be claimed as a dependent`, () => {
    it(`If MFJ and filing for credits, can never be claimed as a dependent`, ({ task }) => {
      task.meta.testedFactPaths = [
        `/primaryFilerIsClaimedAsDependent`,
        `/secondaryFilerIsClaimedAsDependent`,
        `/eitherFilerIsClaimedAsDependent`,
      ];
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/willBeClaimed`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/willBeClaimed`]: createBooleanWrapper(true),
        [`/MFJRequiredToFile`]: createBooleanWrapper(false),
        [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(true),
      });
      expect(factGraph.get(Path.concretePath(`/primaryFilerIsClaimedAsDependent`, null)).get).toBe(false);
      expect(factGraph.get(Path.concretePath(`/secondaryFilerIsClaimedAsDependent`, null)).get).toBe(false);
      expect(factGraph.get(Path.concretePath(`/eitherFilerIsClaimedAsDependent`, null)).get).toBe(false);
    });

    it(`If MFJ and *not* filing for credits, accept the TP's answer of whether someone will claim them`, ({ task }) => {
      task.meta.testedFactPaths = [
        `/primaryFilerIsClaimedAsDependent`,
        `/secondaryFilerIsClaimedAsDependent`,
        `/eitherFilerIsClaimedAsDependent`,
      ];
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
        [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/willBeClaimed`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/willBeClaimed`]: createBooleanWrapper(true),
        [`/MFJRequiredToFile`]: createBooleanWrapper(false),
        [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
      });
      expect(factGraph.get(Path.concretePath(`/primaryFilerIsClaimedAsDependent`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/secondaryFilerIsClaimedAsDependent`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/eitherFilerIsClaimedAsDependent`, null)).get).toBe(true);
    });
  });

  it(`On a return with a single filer, minimumFilerDateOfBirth should return their DOB`, ({ task }) => {
    task.meta.testedFactPaths = [`/minimumFilerDateOfBirth`];

    const birthday = `${Number.parseInt(CURRENT_TAX_YEAR) - 24}-01-02`;
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
      [`/filingStatus`]: createEnumWrapper(`single`, `/filingStatusOptions`),
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(birthday),
    });

    expect(factGraph.get(Path.concretePath(`/minimumFilerDateOfBirth`, null)).get).toStrictEqual(
      DayFactory(birthday).right
    );
  });

  it(`On a MFJ return, minimumFilerDateOfBirth should return the DOB of the older filer`, ({ task }) => {
    task.meta.testedFactPaths = [`/minimumFilerDateOfBirth`];

    const olderFilerBirthday = `${Number.parseInt(CURRENT_TAX_YEAR) - 28}-01-02`;
    const youngerFilerBirthday = `${Number.parseInt(CURRENT_TAX_YEAR) - 24}-01-02`;

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(youngerFilerBirthday),
      [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(olderFilerBirthday),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    });

    expect(factGraph.get(Path.concretePath(`/minimumFilerDateOfBirth`, null)).get).toStrictEqual(
      DayFactory(olderFilerBirthday).right
    );

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(olderFilerBirthday),
      [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(youngerFilerBirthday),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    });

    expect(factGraph2.get(Path.concretePath(`/minimumFilerDateOfBirth`, null)).get).toStrictEqual(
      DayFactory(olderFilerBirthday).right
    );
  });

  it(`A return in which the filer's spouse provides an IP Pin will have that fact recorded`, ({ task }) => {
    task.meta.testedFactPaths = [`/spouseHasIpPin`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/hasIpPin`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/identityPin`]: createIpPinWrapper(`123456`),
    });

    expect(factGraph.get(Path.concretePath(`/spouseHasIpPin`, null)).get).toBe(true);
  });

  it(`A return in which the filer's spouse does not have an IP Pin will have that fact recorded`, ({ task }) => {
    task.meta.testedFactPaths = [`/spouseHasIpPin`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/hasIpPin`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(Path.concretePath(`/spouseHasIpPin`, null)).get).toBe(false);
  });

  it(`A return whose secondary filer has an IP PIN but hasn't
      completed the PIN entry flow should be flagged as missing their PIN`, ({ task }) => {
    task.meta.testedFactPaths = [`/secondaryFilerIsMissingIpPin`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/hasIpPin`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/flowSpouseIpPinReady`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(Path.concretePath(`/secondaryFilerIsMissingIpPin`, null)).get).toBe(true);
  });
  it(`A return whose secondary filer has an IP PIN and has
      completed the PIN entry flow shouldn't be flagged as missing their PIN`, ({ task }) => {
    task.meta.testedFactPaths = [`/secondaryFilerIsMissingIpPin`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/hasIpPin`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/flowSpouseIpPinReady`]: createBooleanWrapper(true),
    });

    expect(factGraph.get(Path.concretePath(`/secondaryFilerIsMissingIpPin`, null)).get).toBe(false);
  });

  it(`A filer who is neither a full-year citizen, a full-year noncitizen resident,
      or a citizen by the end of the year, should be KO'd`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFilerResidencyKnockout`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(Path.concretePath(`/primaryFilerResidencyKnockout`, null)).get).toBe(true);
  });

  it(`A filer who is either a full-year citizen, a full-year noncitizen resident,
      or a citizen by the end of the year, shouldn't be KO'd`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFilerResidencyKnockout`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    });

    expect(factGraph.get(Path.concretePath(`/primaryFilerResidencyKnockout`, null)).get).toBe(false);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(true),
    });

    expect(factGraph2.get(Path.concretePath(`/primaryFilerResidencyKnockout`, null)).get).toBe(false);

    const { factGraph: factGraph3 } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
    });

    expect(factGraph3.get(Path.concretePath(`/primaryFilerResidencyKnockout`, null)).get).toBe(false);
  });

  it(`A non-MFJ filer should be treated as a dependent only if they are a
    dependent and must be claimed by another filer`, ({ task }) => {
    task.meta.testedFactPaths = [`/treatFilersAsDependents`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
      '/filingStatus': createEnumWrapper(`single`, `/filingStatusOptions`),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
      [`/primaryFilerPotentialClaimerFiledOnlyForRefund`]: createBooleanWrapper(true),
    });

    expect(factGraph.get(Path.concretePath(`/treatFilersAsDependents`, null)).get).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
      '/filingStatus': createEnumWrapper(`single`, `/filingStatusOptions`),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
    });

    expect(factGraph2.get(Path.concretePath(`/treatFilersAsDependents`, null)).get).toBe(false);
  });

  it(`A MFJ filer should be treated as a dependent only if they are a
      dependent and must be claimed by another filer`, ({ task }) => {
    task.meta.testedFactPaths = [`/treatFilersAsDependents`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
      [`/writableSeparationAgreement`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
      [`/primaryFilerPotentialClaimerFiledOnlyForRefund`]: createBooleanWrapper(true),
    });

    expect(factGraph.get(Path.concretePath(`/treatFilersAsDependents`, null)).get).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
      [`/writableSeparationAgreement`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
    });

    expect(factGraph2.get(Path.concretePath(`/treatFilersAsDependents`, null)).get).toBe(false);
  });

  it(`Secondary SSN restrictions are known if there is no need to collect the SSN work status`, ({ task }) => {
    task.meta.testedFactPaths = [`/secondaryFilerSocialSecurityNumberRestrictionsAreComplete`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    });

    expect(
      factGraph.get(Path.concretePath(`/secondaryFilerSocialSecurityNumberRestrictionsAreComplete`, null)).get
    ).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
      [`/secondaryFilerSsnEmploymentValidity`]: createEnumWrapper(
        `validOnlyWithDhsAuthorization`,
        `/ssnEmploymentValidityOptions`
      ),
    });

    expect(
      factGraph2.get(Path.concretePath(`/secondaryFilerSocialSecurityNumberRestrictionsAreComplete`, null)).get
    ).toBe(true);
  });

  it(`Secondary filer SSN restrictions are known if the filer has indicated their SSN is valid with restrictions`, ({
    task,
  }) => {
    task.meta.testedFactPaths = [`/secondaryFilerSocialSecurityNumberRestrictionsAreComplete`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
      [`/secondaryFilerSsnEmploymentValidity`]: createEnumWrapper(
        `validOnlyWithDhsAuthorization`,
        `/primaryFilerSsnEmploymentValidityOptions`
      ),
    });

    expect(factGraph.get(`/secondaryFilerSsnEmploymentValidity` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/secondaryFiler/ssnNotValidForEmployment`, null)).get).toBe(false);
    expect(
      factGraph.get(Path.concretePath(`/secondaryFilerSocialSecurityNumberRestrictionsAreComplete`, null)).get
    ).toBe(true);
  });

  it(`Secondary filer SSN restrictions are known if the filer has
      indicated their SSN is not valid for work but can be used for benefits`, ({ task }) => {
    task.meta.testedFactPaths = [`/secondaryFilerSocialSecurityNumberRestrictionsAreComplete`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
      [`/secondaryFilerSsnEmploymentValidity`]: createEnumWrapper(
        `notValid`,
        `/primaryFilerSsnEmploymentValidityOptions`
      ),
    });

    expect(factGraph.get(`/secondaryFilerSsnEmploymentValidity` as ConcretePath).complete).toBe(true);
    expect(
      factGraph.get(Path.concretePath(`/secondaryFilerSocialSecurityNumberRestrictionsAreComplete`, null)).get
    ).toBe(true);
  });

  it(`hasTin and isTinUnique should be true if the filer has provided a TIN and it's unique`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/hasTin`, `/filers/*/isTinUnique`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
    });

    expect(factGraph.get(Path.concretePath(`/primaryFiler/hasTin`, null)).get).toBe(true);
    expect(factGraph.get(Path.concretePath(`/primaryFiler/isTinUnique`, null)).get).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
    });

    expect(factGraph2.get(Path.concretePath(`/primaryFiler/isTinUnique`, null)).get).toBe(false);
  });

  const { factGraph: factGraph3 } = setupFactGraph({});
  expect(factGraph3.get(Path.concretePath(`/primaryFiler/hasTin`, null)).get).toBe(false);
});

describe(`About You`, () => {
  it(`"About You" remains incomplete if filer hasn't indicated whether they are blind`, ({ task }) => {
    task.meta.testedFactPaths = [`/aboutYouIsComplete`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ny`, `/scopedStateOptions`),
      [`/primaryFilerW2And1099IntInScopedState`]: createEnumWrapper(
        `onlySame`,
        `/primaryFilerW2And1099IntStateOptions`
      ),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(`/primaryFilerW2And1099IntInScopedState` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFiler/isBlind` as ConcretePath).complete).toBe(false);
    expect(factGraph.get(`/primaryFilerIsCitizenOrRAAllYear` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/filerResidenceAndIncomeState` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFilerSocialSecurityNumberRestrictionsAreUnderstood` as ConcretePath).complete).toBe(
      true
    );
    expect(
      factGraph.get(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits` as ConcretePath)
        .complete
    ).toBe(true);
    expect(factGraph.get(Path.concretePath(`/aboutYouIsComplete`, null)).get).toBe(false);
  });

  it(`"About You" remains incomplete if the filer hasn't completed their residence & income state info`, ({ task }) => {
    task.meta.testedFactPaths = [`/aboutYouIsComplete`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/isBlind`]: createBooleanWrapper(false),
      [`/primaryFilerW2And1099IntInScopedState`]: createEnumWrapper(
        `onlySame`,
        `/primaryFilerW2And1099IntStateOptions`
      ),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(`/primaryFilerW2And1099IntInScopedState` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFiler/isBlind` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFilerIsCitizenOrRAAllYear` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/filerResidenceAndIncomeState` as ConcretePath).complete).toBe(false);
    expect(factGraph.get(`/primaryFilerSocialSecurityNumberRestrictionsAreUnderstood` as ConcretePath).complete).toBe(
      true
    );
    expect(
      factGraph.get(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits` as ConcretePath)
        .complete
    ).toBe(true);
    expect(factGraph.get(Path.concretePath(`/aboutYouIsComplete`, null)).get).toBe(false);
  });

  it(`"About You" remains incomplete if the filer hasn't indicated in what state(s) they have earned income`, ({
    task,
  }) => {
    task.meta.testedFactPaths = [`/aboutYouIsComplete`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/isBlind`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ny`, `/scopedStateOptions`),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(`/primaryFilerW2And1099IntInScopedState` as ConcretePath).complete).toBe(false);
    expect(factGraph.get(`/primaryFiler/isBlind` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFilerIsCitizenOrRAAllYear` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/filerResidenceAndIncomeState` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFilerSocialSecurityNumberRestrictionsAreUnderstood` as ConcretePath).complete).toBe(
      true
    );
    expect(
      factGraph.get(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits` as ConcretePath)
        .complete
    ).toBe(true);
    expect(factGraph.get(Path.concretePath(`/aboutYouIsComplete`, null)).get).toBe(false);
  });

  it(`"About You" remains incomplete if the filer has not indicated whether they can be claimed by another filer`, ({
    task,
  }) => {
    task.meta.testedFactPaths = [`/aboutYouIsComplete`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/isBlind`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ny`, `/scopedStateOptions`),
      [`/primaryFilerW2And1099IntInScopedState`]: createEnumWrapper(
        `onlySame`,
        `/primaryFilerW2And1099IntStateOptions`
      ),
    });

    expect(factGraph.get(`/primaryFilerW2And1099IntInScopedState` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFiler/isBlind` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFilerIsCitizenOrRAAllYear` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/filerResidenceAndIncomeState` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFilerSocialSecurityNumberRestrictionsAreUnderstood` as ConcretePath).complete).toBe(
      true
    );
    expect(
      factGraph.get(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits` as ConcretePath)
        .complete
    ).toBe(false);
    expect(factGraph.get(Path.concretePath(`/aboutYouIsComplete`, null)).get).toBe(false);
  });

  it(`"About You" remains incomplete if the filer hasn't given their full citizenship/residency info`, ({ task }) => {
    task.meta.testedFactPaths = [`/aboutYouIsComplete`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/isBlind`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ny`, `/scopedStateOptions`),
      [`/primaryFilerW2And1099IntInScopedState`]: createEnumWrapper(
        `onlySame`,
        `/primaryFilerW2And1099IntStateOptions`
      ),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(`/primaryFilerW2And1099IntInScopedState` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFiler/isBlind` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFilerIsCitizenOrRAAllYear` as ConcretePath).complete).toBe(false);
    expect(factGraph.get(`/filerResidenceAndIncomeState` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFilerSocialSecurityNumberRestrictionsAreUnderstood` as ConcretePath).complete).toBe(
      true
    );
    expect(
      factGraph.get(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits` as ConcretePath)
        .complete
    ).toBe(true);
    expect(factGraph.get(Path.concretePath(`/aboutYouIsComplete`, null)).get).toBe(false);
  });

  it(`"About You" is registered complete when all required sections are complete`, ({ task }) => {
    task.meta.testedFactPaths = [`/aboutYouIsComplete`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/firstName`]: createStringWrapper(`Test`),
      [`/filers/#${primaryFilerId}/lastName`]: createStringWrapper(`Testerson`),
      [`/address`]: createAddressWrapper(`742 Evergreen Terrace`, `Springfield`, `12345`, `CA`),
      [`/phone`]: createPhoneWrapper(`789`, `333`, `1234`),
      [`/filers/#${primaryFilerId}/isBlind`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/isDisabled`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/isStudent`]: createBooleanWrapper(false),
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ny`, `/scopedStateOptions`),
      [`/primaryFilerW2And1099IntInScopedState`]: createEnumWrapper(
        `onlySame`,
        `/primaryFilerW2And1099IntStateOptions`
      ),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
      [`/receivedAlaskaPfd`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(`/primaryFilerW2And1099IntInScopedState` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFiler/isBlind` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFilerIsCitizenOrRAAllYear` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/filerResidenceAndIncomeState` as ConcretePath).complete).toBe(true);
    expect(factGraph.get(`/primaryFilerSocialSecurityNumberRestrictionsAreUnderstood` as ConcretePath).complete).toBe(
      true
    );
    expect(
      factGraph.get(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits` as ConcretePath)
        .complete
    ).toBe(true);
    expect(factGraph.get(Path.concretePath(`/aboutYouIsComplete`, null)).get).toBe(true);
  });
});

describe(`HoH eligibility`, () => {
  it(`A married person is eligible for HoH if they lived separately for 6 months`, ({ task }) => {
    task.meta.testedFactPaths = [`/filerCouldQualifyForHoh`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
      [`/spouseLivedTogetherMonths`]: createEnumWrapper(
        `livedTogetherSixMonthsOrLess`,
        `/spouseLivedTogetherMonthsOptions`
      ),
      [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    });
    expect(factGraph.get(Path.concretePath(`/eligibleForMFJ`, null)).get).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filerCouldQualifyForHoh`, null)).get).toBe(true);
  });

  it(`A married person is eligible for HoH if their spouse was a nonresident alien
     and they are a US Citizen or Resident`, ({ task }) => {
    task.meta.testedFactPaths = [`/filerCouldQualifyForHoh`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(false),
    });
    expect(factGraph.get(Path.concretePath(`/secondaryFilerResidencyEligibleForMFJ`, null)).get).toBe(false);
    expect(factGraph.get(Path.concretePath(`/eligibleForMFJ`, null)).get).toBe(false);
    expect(factGraph.get(Path.concretePath(`/filerCouldQualifyForHoh`, null)).get).toBe(true);
  });
  it(`A married person is _not_ eligible for HoH if they lived together but had a separation agreement`, ({ task }) => {
    task.meta.testedFactPaths = [`/filerCouldQualifyForHoh`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
      [`/writableSeparationAgreement`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    });
    expect(factGraph.get(Path.concretePath(`/eligibleForMFJ`, null)).get).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filerCouldQualifyForHoh`, null)).get).toBe(false);
  });
  it(`A single or divorced person is eligible for HoH`, ({ task }) => {
    task.meta.testedFactPaths = [`/filerCouldQualifyForHoh`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
    });
    expect(factGraph.get(Path.concretePath(`/filerCouldQualifyForHoh`, null)).get).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`divorced`, `/maritalStatusOptions`),
    });
    expect(factGraph2.get(Path.concretePath(`/filerCouldQualifyForHoh`, null)).get).toBe(true);
  });

  describe(`Spouse and Marital Status tests`, () => {
    it(`The return of a filer who indicates they are married will reflect that fact`, ({ task }) => {
      task.meta.testedFactPaths = [`/isMarried`];

      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      });

      expect(factGraph.get(Path.concretePath(`/isMarried`, null)).get).toBe(true);
    });

    it(`The return of a filer who indicates they are single will not reflect that they are married`, ({ task }) => {
      task.meta.testedFactPaths = [`/isMarried`];

      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
      });

      expect(factGraph.get(Path.concretePath(`/isMarried`, null)).get).toBe(false);
    });

    it(`The return of a filer who indicates they lived apart from their
        spouse for the entire year will reflect that fact`, ({ task }) => {
      task.meta.testedFactPaths = [`/livedApartAllYear`];
      const { factGraph } = setupFactGraph({
        [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
        '/writableLivedApartLastSixMonths': createBooleanWrapper(true),
        [`/spouseLivedTogetherMonths`]: createEnumWrapper(`livedApartAllYear`, `/spouseLivedTogetherMonthsOptions`),
      });

      expect(factGraph.get(Path.concretePath(`/livedApartAllYear`, null)).get).toBe(true);
    });

    it(`The return of a filer who indicates they lived apart from their spouse for 6 months
        but not the entire year will reflect that fact`, ({ task }) => {
      task.meta.testedFactPaths = [`/livedApartAllYear`];
      const { factGraph } = setupFactGraph({
        [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
        '/writableLivedApartLastSixMonths': createBooleanWrapper(true),
        [`/spouseLivedTogetherMonths`]: createEnumWrapper(
          `livedTogetherSixMonthsOrLess`,
          `/spouseLivedTogetherMonthsOptions`
        ),
      });

      expect(factGraph.get(Path.concretePath(`/livedApartAllYear`, null)).get).toBe(false);
    });

    it(`The return of a filer who indicates they are widowed will reflect that fact`, ({ task }) => {
      task.meta.testedFactPaths = [`/isWidowed`];

      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      });

      expect(factGraph.get(Path.concretePath(`/isWidowed`, null)).get).toBe(true);
    });

    it(`The return of a filer whose spouse passed away in the tax year will reflect that fact`, ({ task }) => {
      task.meta.testedFactPaths = [`/isWidowedInTaxYear`];

      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
      });

      expect(factGraph.get(Path.concretePath(`/isWidowedInTaxYear`, null)).get).toBe(true);

      const { factGraph: factGraph2 } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusOne`, `/yearOfSpouseDeathOptions`),
      });

      expect(factGraph2.get(Path.concretePath(`/isWidowedInTaxYear`, null)).get).toBe(false);
    });

    it(`The return of a filer whose spouse passed away more than two years before
        the current tax year will reflect that fact`, ({ task }) => {
      task.meta.testedFactPaths = [`/widowedBeforeTaxYearMinusTwo`];

      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`beforeTaxYearMinusTwo`, `/yearOfSpouseDeathOptions`),
      });

      expect(factGraph.get(Path.concretePath(`/widowedBeforeTaxYearMinusTwo`, null)).get).toBe(true);

      const { factGraph: factGraph2 } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusOne`, `/yearOfSpouseDeathOptions`),
      });

      expect(factGraph2.get(Path.concretePath(`/widowedBeforeTaxYearMinusTwo`, null)).get).toBe(false);
    });

    it(`A widowed filer whose spouse passed away in the year or two before the tax year
        could qualify for QSS`, ({ task }) => {
      task.meta.testedFactPaths = [`/widowedCouldQualifyForQSS`];

      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusTwo`, `/yearOfSpouseDeathOptions`),
      });

      expect(factGraph.get(Path.concretePath(`/widowedCouldQualifyForQSS`, null)).get).toBe(true);

      const { factGraph: factGraph2 } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusOne`, `/yearOfSpouseDeathOptions`),
      });

      expect(factGraph2.get(Path.concretePath(`/widowedCouldQualifyForQSS`, null)).get).toBe(true);
    });

    it(`A widowed filer whose spouse passed away more than two before the tax year
        no longer qualifies for QSS`, ({ task }) => {
      task.meta.testedFactPaths = [`/widowedCouldQualifyForQSS`];

      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`beforeTaxYearMinusTwo`, `/yearOfSpouseDeathOptions`),
      });

      expect(factGraph.get(Path.concretePath(`/widowedCouldQualifyForQSS`, null)).get).toBe(false);
    });
  });

  it(`A widowed person is eligible for HoH`, ({ task }) => {
    task.meta.testedFactPaths = [`/filerCouldQualifyForHoh`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusOne`, `/yearOfSpouseDeathOptions`),
    });
    expect(factGraph.get(Path.concretePath(`/filerCouldQualifyForHoh`, null)).get).toBe(true);
  });

  it(`A widowed person is ineligible for HoH if their spouse died during the tax year and they were not separated`, ({
    task,
  }) => {
    task.meta.testedFactPaths = [`/filerCouldQualifyForHoh`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
      [`/canFileJointlyYearOfSpouseDeath`]: createBooleanWrapper(true),
      [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    });
    expect(factGraph.get(Path.concretePath(`/maritalStatusAllowsFilingMarried`, null)).get).toBe(true);
    expect(factGraph.get(Path.concretePath(`/eligibleForMFJ`, null)).get).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filerCouldQualifyForHoh`, null)).get).toBe(false);
  });
});

describe(`MFJ dependents`, () => {
  it(`If they have chosen to file as MFJ, either of them can be claimed, and they opt to not file for credits`, ({
    task,
  }) => {
    task.meta.testedFactPaths = [`/isMFJDependent`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/MFJRequiredToFile`]: createBooleanWrapper(false),
      [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
    });
    expect(factGraph.get(Path.concretePath(`/isMFJDependent`, null)).get).toBe(true);
  });

  it(`MFJ filers are not dependents if they are eligible for MFJ
        and can't be claimed by another filer`, ({ task }) => {
    task.meta.testedFactPaths = [`/isMFJDependent`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
      [`/MFJRequiredToFile`]: createBooleanWrapper(true),
      [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
    });
    expect(factGraph.get(Path.concretePath(`/isMFJDependent`, null)).get).toBe(false);
  });

  it(`The return of filers who have chosen to be treated as MFJ dependents and are filing
        in order to get a refund should reflect that fact`, ({ task }) => {
    task.meta.testedFactPaths = [`/MFJClaimingRefundOnly`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
    });
    expect(factGraph.get(Path.concretePath(`/MFJClaimingRefundOnly`, null)).get).toBe(true);
  });

  it(`The return of filers who have chosen to be treated as MFJ dependents and are filing
        for credits should reflect that they are not only filing for a refund`, ({ task }) => {
    task.meta.testedFactPaths = [`/MFJClaimingRefundOnly`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
      [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(true),
    });
    expect(factGraph.get(Path.concretePath(`/MFJClaimingRefundOnly`, null)).get).toBe(false);
  });

  it(`A MFJ filer should see the Spouse Filing Requirement Subsection if either
      the primary secondary filer can be claimed as a dependent, `, ({ task }) => {
    task.meta.testedFactPaths = [`/flowMFJSpouseFilingRequirementSubsection`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
    });
    expect(factGraph.get(Path.concretePath(`/flowMFJSpouseFilingRequirementSubsection`, null)).get).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
    });
    expect(factGraph2.get(Path.concretePath(`/flowMFJSpouseFilingRequirementSubsection`, null)).get).toBe(true);
  });

  it(`A MFJ filer shouldn't see the Spouse Filing Requirement Subsection if neither
      the primary secondary filer can be claimed as a dependent, `, ({ task }) => {
    task.meta.testedFactPaths = [`/flowMFJSpouseFilingRequirementSubsection`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
    });
    expect(factGraph.get(Path.concretePath(`/flowMFJSpouseFilingRequirementSubsection`, null)).get).toBe(false);
  });
});

describe(`QSS eligibility`, () => {
  it(`A filer is eligible for QSS if their spouse died 1 or 2 years ago`, ({ task }) => {
    task.meta.testedFactPaths = [`/filerCouldQualifyForQss`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusOne`, `/yearOfSpouseDeathOptions`),
      [`/canFileJointlyYearOfSpouseDeath`]: createBooleanWrapper(true),
    });
    expect(factGraph.get(Path.concretePath(`/filerCouldQualifyForQss`, null)).get).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusTwo`, `/yearOfSpouseDeathOptions`),
      [`/canFileJointlyYearOfSpouseDeath`]: createBooleanWrapper(true),
    });
    expect(factGraph2.get(Path.concretePath(`/filerCouldQualifyForQss`, null)).get).toBe(true);
  });

  it(`A filer is not eligible for QSS if their spouse died within the tax year`, ({ task }) => {
    task.meta.testedFactPaths = [`/filerCouldQualifyForQss`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
    });
    expect(factGraph.get(Path.concretePath(`/filerCouldQualifyForQss`, null)).get).toBe(false);
  });

  it(`A filer is ineligible for QSS if their spouse died more than 2 years ago`, ({ task }) => {
    task.meta.testedFactPaths = [`/filerCouldQualifyForQss`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`beforeTaxYearMinusTwo`, `/yearOfSpouseDeathOptions`),
    });
    expect(factGraph.get(Path.concretePath(`/filerCouldQualifyForQss`, null)).get).toBe(false);
  });

  it(`A filer cannot qualify for QSS if they were unable to file jointly the year of their spouse's death`, ({
    task,
  }) => {
    task.meta.testedFactPaths = [`/filerCouldQualifyForQss`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusTwo`, `/yearOfSpouseDeathOptions`),
      [`/canFileJointlyYearOfSpouseDeath`]: createBooleanWrapper(false),
    });
    expect(factGraph.get(Path.concretePath(`/filerCouldQualifyForQss`, null)).get).toBe(false);
  });
});

describe(`MFJ eligibility`, () => {
  it(`A filer is eligible for MFJ if they were married and their spouse is a citizen or resident alien`, ({ task }) => {
    task.meta.testedFactPaths = [`/secondaryFilerResidencyEligibleForMFJ`, `/eligibleForMFJ`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
      [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    });
    expect(factGraph.get(Path.concretePath(`/secondaryFilerResidencyEligibleForMFJ`, null)).get).toBe(true);
    expect(factGraph.get(Path.concretePath(`/eligibleForMFJ`, null)).get).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
      [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(true),
    });
    expect(factGraph2.get(Path.concretePath(`/secondaryFilerResidencyEligibleForMFJ`, null)).get).toBe(true);
    expect(factGraph2.get(Path.concretePath(`/eligibleForMFJ`, null)).get).toBe(true);

    const { factGraph: factGraph3 } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
      [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
    });
    expect(factGraph3.get(Path.concretePath(`/secondaryFilerResidencyEligibleForMFJ`, null)).get).toBe(true);
    expect(factGraph3.get(Path.concretePath(`/eligibleForMFJ`, null)).get).toBe(true);
  });

  it(`A filer is not eligible for MFJ if they were married and their spouse is a nonresident alien`, ({ task }) => {
    task.meta.testedFactPaths = [`/secondaryFilerResidencyEligibleForMFJ`, `/eligibleForMFJ`];
    const { factGraph: factGraph3 } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
      [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(false),
    });
    expect(factGraph3.get(Path.concretePath(`/secondaryFilerResidencyEligibleForMFJ`, null)).get).toBe(false);
    expect(factGraph3.get(Path.concretePath(`/eligibleForMFJ`, null)).get).toBe(false);
  });
});

describe(`MFS spouse`, () => {
  // for MFSSpouseFilingReturnDerived tests
  type MFSSpouseFilingReturnDerivedCases = {
    name: string;
    maritalStatus: string; // should be enum but lazy
    livingFiling?: boolean;
    deceasedFiling?: boolean;
    expected?: boolean;
  };
  type MFSSpouseFilingReturnWritables = {
    [path: string]: FactValue;
  };

  const setupFilingFacts = (testCase: MFSSpouseFilingReturnDerivedCases) => {
    const facts: MFSSpouseFilingReturnWritables = {};

    if (Object.hasOwn(testCase, `livingFiling`)) {
      facts[`/MFSLivingSpouseFilingReturn`] = createBooleanWrapper(testCase.livingFiling as boolean);
    }
    if (Object.hasOwn(testCase, `deceasedFiling`)) {
      facts[`/MFSDeceasedSpouseFilingReturn`] = createBooleanWrapper(testCase.deceasedFiling as boolean);
    }

    return facts;
  };

  it(`The MFS spouse provides no benefits if they are filing a return`, ({ task }) => {
    task.meta.testedFactPaths = [`/MFSSpouseProvidesNoBenefits`];
    const { factGraph } = setupFactGraph({
      [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(true),
    });
    expect(factGraph.get(Path.concretePath(`/MFSSpouseProvidesNoBenefits`, null)).get).toBe(true);
  });

  it(`The MFS spouse provides no benefits if they have gross income`, ({ task }) => {
    task.meta.testedFactPaths = [`/MFSSpouseProvidesNoBenefits`];
    const { factGraph } = setupFactGraph({
      [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(false),
      [`/MFSSpouseHasGrossIncome`]: createBooleanWrapper(true),
    });

    expect(factGraph.get(Path.concretePath(`/MFSSpouseProvidesNoBenefits`, null)).get).toBe(true);
  });

  it(`The MFS spouse provides no benefits if they can be claimed`, ({ task }) => {
    task.meta.testedFactPaths = [`/MFSSpouseProvidesNoBenefits`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(false),
      [`/MFSSpouseHasGrossIncome`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
    });
    expect(factGraph.get(Path.concretePath(`/secondaryFiler/canBeClaimed`, null)).get).toBe(true);
    expect(factGraph.get(Path.concretePath(`/MFSSpouseProvidesNoBenefits`, null)).get).toBe(true);
  });

  it(`The MFS spouse provides benefits if they can't claimed, have no income, and are not filing`, ({ task }) => {
    task.meta.testedFactPaths = [`/MFSSpouseProvidesNoBenefits`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
      [`/MFSSpouseHasGrossIncome`]: createBooleanWrapper(false),
      [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(Path.concretePath(`/MFSSpouseProvidesNoBenefits`, null)).get).toBe(false);
  });

  it(`If the MFS spouse itemizes deductions, filer should be KO'd`, ({ task }) => {
    task.meta.testedFactPaths = [`/spouseItemizesKnockout`];
    const { factGraph } = setupFactGraph({
      [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(true),
      [`/spouseItemizes`]: createBooleanWrapper(true),
    });

    expect(factGraph.get(Path.concretePath(`/spouseItemizesKnockout`, null)).get).toBe(true);
  });

  it(`If the MFS spouse does not itemize deductions,
      filer should not be KO'd`, ({ task }) => {
    task.meta.testedFactPaths = [`/spouseItemizesKnockout`];
    const { factGraph } = setupFactGraph({
      [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(true),
      [`/spouseItemizes`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(Path.concretePath(`/spouseItemizesKnockout`, null)).get).toBe(false);
  });

  it(`If the MFS spouse is a citizen or has a NRTIN, the filer
      needs to provide their TIN`, ({ task }) => {
    task.meta.testedFactPaths = [`/MFSNeedSpouseTaxId`];
    const { factGraph } = setupFactGraph({
      [`/MFSSpouseHasNRTIN`]: createBooleanWrapper(true),
    });
    expect(factGraph.get(Path.concretePath(`/MFSNeedSpouseTaxId`, null)).get).toBe(true);
    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    });
    expect(factGraph2.get(Path.concretePath(`/MFSNeedSpouseTaxId`, null)).get).toBe(true);
  });

  it(`If the MFS spouse is not a citizen and does not have a NRTIN, the filer
      does not need to provide a TIN`, ({ task }) => {
    task.meta.testedFactPaths = [`/MFSNeedSpouseTaxId`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/MFSSpouseHasNRTIN`]: createBooleanWrapper(false),
    });
    expect(factGraph.get(Path.concretePath(`/MFSNeedSpouseTaxId`, null)).get).toBe(false);
  });

  describe(`MFSSpouseFilingReturnDerived living v deceased cases`, () => {
    for (const testCase of [
      //happy path cases
      {
        name: `Married, true happy path should derive to true`,
        maritalStatus: `married`,
        livingFiling: true,
        expected: true,
      },
      {
        name: `Widowed, true happy path should derive to true`,
        maritalStatus: `widowed`,
        deceasedFiling: true,
        expected: true,
      },
      {
        name: `Married, false happy path should derive to false`,
        maritalStatus: `married`,
        livingFiling: false,
        expected: false,
      },
      {
        name: `Widowed, false happy path should derive to false`,
        maritalStatus: `widowed`,
        deceasedFiling: false,
        expected: false,
      },

      // set marital status, both facts true -> true
      {
        name: `Married, both living and widowed are true should derive to true`,
        maritalStatus: `married`,
        livingFiling: true,
        deceasedFiling: true,
        expected: true,
      },
      {
        name: `Widowed, both living and widowed are true should derive to true`,
        maritalStatus: `widowed`,
        livingFiling: true,
        deceasedFiling: true,
        expected: true,
      },

      // set marital status, correct fact false, other fact true -> false
      {
        name: `Married, living fact false, widowed fact true should derive to false`,
        maritalStatus: `married`,
        livingFiling: false,
        deceasedFiling: true,
        expected: false,
      },
      {
        name: `Widowed, living fact true, widowed false should derive to false`,
        maritalStatus: `widowed`,
        livingFiling: true,
        deceasedFiling: false,
        expected: false,
      },

      // other marital statuses should derive to true but will be immaterial
      // in the real world, unless the TP switches to MFS
      {
        name: `Single filers with living fact set should derive to false`,
        maritalStatus: `single`,
        livingFiling: true,
        expected: true,
      },
      {
        name: `Divorced filers with living fact set should derive to false`,
        maritalStatus: `divorced`,
        livingFiling: true,
        expected: true,
      },
    ]) {
      it(testCase.name, ({ task }) => {
        task.meta.testedFactPaths = [
          `/MFSSpouseFilingReturnDerived`,
          `/MFSLivingSpouseFilingReturn`,
          `/MFSDeceasedSpouseFilingReturn`,
          `/MFSSpouseFilingReturn`,
        ];

        const { factGraph } = setupFactGraph({
          [`/maritalStatus`]: createEnumWrapper(testCase.maritalStatus, `/maritalStatusOptions`),
          ...setupFilingFacts(testCase),
        });

        expect(factGraph.get(Path.concretePath(`/MFSSpouseFilingReturnDerived`, null)).get).toBe(testCase.expected);
      });
    }
  });

  describe(`Having legacy /MFSSpouseFilingReturn be set'`, () => {
    it(`if true, /MFSSpouseFilingReturnDerived is true`, ({ task }) => {
      task.meta.testedFactPaths = [`/MFSSpouseFilingReturnDerived`, `/MFSSpouseFilingReturn`];

      const { factGraph } = setupFactGraph({
        [`/MFSSpouseFilingReturn`]: createBooleanWrapper(true),
      });

      expect(factGraph.get(Path.concretePath(`/MFSSpouseFilingReturnDerived`, null)).get).toBe(true);
    });
    it(`if false, /MFSSpouseFilingReturnDerived is false`, ({ task }) => {
      task.meta.testedFactPaths = [`/MFSSpouseFilingReturnDerived`, `/MFSSpouseFilingReturn`];

      const { factGraph } = setupFactGraph({
        [`/MFSSpouseFilingReturn`]: createBooleanWrapper(false),
      });

      expect(factGraph.get(Path.concretePath(`/MFSSpouseFilingReturnDerived`, null)).get).toBe(false);
    });

    // if NEW and DEPRECATED facts are both set, default to NEW
    it(`if /MFSSpouseFilingReturn is set _and_ one of the \
      /MFS[Living|Deceased]FilingReturn is set, default to \
      /MFS[Living|Deceased]FilingReturn`, ({ task }) => {
      task.meta.testedFactPaths = [`/MFSSpouseFilingReturnDerived`, `/MFSSpouseFilingReturn`];

      const { factGraph } = setupFactGraph({
        [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
        [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(false),
        [`/MFSSpouseFilingReturn`]: createBooleanWrapper(true),
      });

      expect(factGraph.get(Path.concretePath(`/MFSSpouseFilingReturnDerived`, null)).get).toBe(false);
    });
  });

  describe(`MFSSpouseFilingReturnDerived incomplete cases`, () => {
    for (const testCase of [
      // set marital status, incorrect fact true -> incomplete
      {
        name: `Married without living fact set should be incomplete`,
        maritalStatus: `married`,
        deceasedFiling: true,
      },
      {
        name: `Widowed without widowed fact set should be incomplete`,
        maritalStatus: `widowed`,
        livingFiling: true,
      },
      {
        name: `Single filers with deceased fact set should derive to false`,
        maritalStatus: `single`,
        deceasedFiling: true,
      },
      {
        name: `Divorced filers with deceased fact set should derive to false`,
        maritalStatus: `divorced`,
        deceasedFiling: true,
      },
    ]) {
      it(testCase.name, ({ task }) => {
        task.meta.testedFactPaths = [
          `/MFSSpouseFilingReturnDerived`,
          `/MFSLivingSpouseFilingReturn`,
          `/MFSDeceasedSpouseFilingReturn`,
          `/MFSSpouseFilingReturn`,
        ];

        const { factGraph } = setupFactGraph({
          [`/maritalStatus`]: createEnumWrapper(testCase.maritalStatus, `/maritalStatusOptions`),
          ...setupFilingFacts(testCase),
        });

        expect(factGraph.get(`/MFSSpouseFilingReturnDerived` as ConcretePath).complete).toBe(false);
      });
    }
  });
});

describe(`EITC eligibility`, () => {
  describe(`Single or MFS`, () => {
    it(`Must be a citizen or RA all year`, ({ task }) => {
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];
      // this test is a little silly at the moment, because the filer would get knocked out of direct file
      // due to scope but maybe it will help someone in TY2024.
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(false),
      });
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(false);
    });
    it(`Being a noncitizen, nonresident US National does not qualify the person for EITC`, ({ task }) => {
      // this test is a little silly at the moment, because the filer would get knocked out of direct file
      // due to scope but maybe it will help someone in TY2024.
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNational`]: createBooleanWrapper(true),
      });
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(false);
    });
    it(`US citizens for the full year qualify for EITC with no additional SSN facts`, ({ task }) => {
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      });
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(true);
    });

    it(`US residents for the full year do not get EITC if they had an ITIN instead of an SSN`, ({ task }) => {
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `999`, group: `11`, serial: `1111` }),
      });
      expect(factGraph.get(Path.concretePath(`/filers/*/tin/isSSN`, primaryFilerId)).get).toBe(false);
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(false);
    });

    it(`US residents for the full year qualify for EITC if their SSN Work status is valid`, ({ task }) => {
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/primaryFilerSsnEmploymentValidity`]: createEnumWrapper(
          `validOnlyWithDhsAuthorization`,
          `/primaryFilerSsnEmploymentValidityOptions`
        ),
      });
      expect(factGraph.get(Path.concretePath(`/filersHaveValidSSNsForEitc`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/filersHaveValidSSNsForEitc`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(true);
    });

    // Skipped because there's a bug!
    it(`US residents for the full year qualify for EITC if their SSN Work status is
     invalid, but not just for federally funded benefits`, ({ task }) => {
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/primaryFilerSsnEmploymentValidity`]: createEnumWrapper(
          `notValid`,
          `/primaryFilerSsnEmploymentValidityOptions`
        ),
        [`/filers/#${primaryFilerId}/writableHasSSNOnlyForBenefits`]: createBooleanWrapper(false),
      });
      expect(factGraph.get(Path.concretePath(`/filersHaveValidSSNsForEitc`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(true);

      // There's a bug in the fact dictionary here!!!!
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];
      const { factGraph: factGraph2 } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/primaryFilerSsnEmploymentValidity`]: createEnumWrapper(
          `notValid`,
          `/primaryFilerSsnEmploymentValidityOptions`
        ),
        [`/filers/#${primaryFilerId}/writableHasSSNOnlyForBenefits`]: createBooleanWrapper(true),
      });
      expect(factGraph2.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(false);
    });
  });

  describe(`MFJ`, () => {
    it(`Filers are ineligible if they've chosen to be treated as MFJ dependents`, ({ task }) => {
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
        [`/MFJRequiredToFile`]: createBooleanWrapper(false),
        [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
      });
      expect(factGraph.get(Path.concretePath(`/isMFJDependent`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(false);
    });

    it(`EITC eligible when both filers have SSNs, cant be claimed, and are citizens`, ({ task }) => {
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
      });
      expect(factGraph.get(Path.concretePath(`/isMFJDependent`, null)).get).toBe(false);
      expect(factGraph.get(Path.concretePath(`/filersUSCitizenOrRAAllYear`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/filersHaveValidSSNsForEitc`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(true);
    });

    it(`not eligible when either filer has an ITIN`, ({ task }) => {
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        // the below is an ITIN
        [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `999`, group: `11`, serial: `1111` }),
        [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
      });
      expect(factGraph.get(Path.concretePath(`/isMFJDependent`, null)).get).toBe(false);
      expect(factGraph.get(Path.concretePath(`/filersUSCitizenOrRAAllYear`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/filersHaveValidSSNsForEitc`, null)).get).toBe(false);
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(false);
    });

    it(`not eligible when either filer has a work restriction on their SSN`, ({ task }) => {
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/secondaryFilerSsnEmploymentValidity`]: createEnumWrapper(
          `validOnlyWithDhsAuthorizationExpired`,
          `/ssnEmploymentValidityOptions`
        ),
      });
      expect(factGraph.get(Path.concretePath(`/filersUSCitizenOrRAAllYear`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/secondaryFiler/ssnValidForEitc`, null)).get).toBe(false);
      expect(factGraph.get(Path.concretePath(`/filersHaveValidSSNsForEitc`, null)).get).toBe(false);
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(false);

      const { factGraph: factGraph2 } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
        [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/primaryFilerSsnEmploymentValidity`]: createEnumWrapper(
          `validOnlyWithDhsAuthorizationExpired`,
          `/primaryFilerSsnEmploymentValidityOptions`
        ),
      });
      expect(factGraph2.get(Path.concretePath(`/filersUSCitizenOrRAAllYear`, null)).get).toBe(true);
      expect(factGraph2.get(Path.concretePath(`/primaryFiler/ssnValidForEitc`, null)).get).toBe(false);
      expect(factGraph2.get(Path.concretePath(`/filersHaveValidSSNsForEitc`, null)).get).toBe(false);
      expect(factGraph2.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(false);
    });

    it(`not eligible when primary filer received their SSN for federal benefits`, ({ task }) => {
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];

      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
        [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/primaryFilerSsnEmploymentValidity`]: createEnumWrapper(
          `notValid`,
          `/primaryFilerSsnEmploymentValidityOptions`
        ),
        [`/filers/#${primaryFilerId}/writableHasSSNOnlyForBenefits`]: createBooleanWrapper(true),
      });
      expect(factGraph.get(Path.concretePath(`/filersUSCitizenOrRAAllYear`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/primaryFiler/ssnValidForEitc`, null)).get).toBe(false);
      expect(factGraph.get(Path.concretePath(`/filersHaveValidSSNsForEitc`, null)).get).toBe(false);
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(false);

      const { factGraph: factGraph2 } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
        [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/primaryFilerSsnEmploymentValidity`]: createEnumWrapper(
          `notValid`,
          `/primaryFilerSsnEmploymentValidityOptions`
        ),
        [`/filers/#${primaryFilerId}/writableHasSSNOnlyForBenefits`]: createBooleanWrapper(false),
      });
      expect(factGraph2.get(Path.concretePath(`/filersUSCitizenOrRAAllYear`, null)).get).toBe(true);
      expect(factGraph2.get(Path.concretePath(`/primaryFiler/ssnValidForEitc`, null)).get).toBe(true);
      expect(factGraph2.get(Path.concretePath(`/filersHaveValidSSNsForEitc`, null)).get).toBe(true);
      expect(factGraph2.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(true);
    });

    it(`not eligible when secondary filer received their SSN for federal benefits`, ({ task }) => {
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];

      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/secondaryFilerSsnEmploymentValidity`]: createEnumWrapper(`notValid`, `/ssnEmploymentValidityOptions`),
        [`/filers/#${spouseId}/writableHasSSNOnlyForBenefits`]: createBooleanWrapper(true),
      });
      expect(factGraph.get(Path.concretePath(`/filersUSCitizenOrRAAllYear`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/filersHaveValidSSNsForEitc`, null)).get).toBe(false);
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(false);

      const { factGraph: factGraph2 } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
        [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
        [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
        [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `111`, group: `11`, serial: `1111` }),
        [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
        [`/secondaryFilerSsnEmploymentValidity`]: createEnumWrapper(`notValid`, `/ssnEmploymentValidityOptions`),
        [`/filers/#${spouseId}/writableHasSSNOnlyForBenefits`]: createBooleanWrapper(false),
      });
      expect(factGraph2.get(Path.concretePath(`/filersUSCitizenOrRAAllYear`, null)).get).toBe(true);
      expect(factGraph2.get(Path.concretePath(`/filersHaveValidSSNsForEitc`, null)).get).toBe(true);
      expect(factGraph2.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(true);
    });
  });

  describe(`MFS`, () => {
    it(`Must have lived apart from their spouse for six months or have a separation agreement`, ({ task }) => {
      task.meta.testedFactPaths = [`/filersCouldQualifyForEitc`];
      const { factGraph } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
        [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
      });
      expect(factGraph.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(false);

      const { factGraph: factGraph2 } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
        [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
        [`/spouseLivedTogetherMonths`]: createEnumWrapper(
          `livedTogetherSixMonthsOrLess`,
          `/spouseLivedTogetherMonthsOptions`
        ),
        [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
      });

      expect(factGraph2.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(true);

      const { factGraph: factGraph3 } = setupFactGraph({
        ...baseFilersData,
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatusChoice': createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
        [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
        [`/spouseLivedTogetherMonths`]: createEnumWrapper(
          `livedTogetherSixMonthsOrLess`,
          `/spouseLivedTogetherMonthsOptions`
        ),
        '/writableLivedApartLastSixMonths': createBooleanWrapper(false),
        '/writableSeparationAgreement': createBooleanWrapper(true),
      });
      expect(factGraph3.get(Path.concretePath(`/filersCouldQualifyForEitc`, null)).get).toBe(true);
    });
  });
});

describe(`EITC Age tests`, () => {
  it(`Someone who turns 25 on January 1 of the next tax year counts as 25 for EITC`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/age25OrOlderForEitc`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 24}-01-01`),
    });
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age`, null)).get).toBe(24);
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age25OrOlderForEitc`, null)).get).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 24}-01-02`),
    });
    expect(factGraph2.get(Path.concretePath(`/primaryFiler/age`, null)).get).toBe(24);
    expect(factGraph2.get(Path.concretePath(`/primaryFiler/age25OrOlderForEitc`, null)).get).toBe(false);
  });

  it(`A 21 year old is below 25`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/age25OrOlderForEitc`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 21}-06-01`),
    });
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age`, null)).get).toBe(21);
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age25OrOlderForEitc`, null)).get).toBe(false);
  });

  it(`A 27 year old is above 25`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/age25OrOlderForEitc`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 27}-06-01`),
    });
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age`, null)).get).toBe(27);
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age25OrOlderForEitc`, null)).get).toBe(true);
  });

  it(`Someone who turns 65 on January 1 of the next tax year is 64 for EITC`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/age64OrYoungerForEitc`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 64}-01-01`),
    });
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age`, null)).get).toBe(64);
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age64OrYoungerForEitc`, null)).get).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 65}-12-31`),
    });
    expect(factGraph2.get(Path.concretePath(`/primaryFiler/age`, null)).get).toBe(65);
    expect(factGraph2.get(Path.concretePath(`/primaryFiler/age64OrYoungerForEitc`, null)).get).toBe(false);
  });

  it(`A spouse who died the day before their 65th birthday is eligible for EITC`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/age64OrYoungerForEitc`];
    // If they die on their birthday, no EITC for them. rough.
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
      [`/secondaryFilerDateOfDeath`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR)}-06-01`),
      [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 65}-06-01`),
    });
    expect(factGraph.get(Path.concretePath(`/secondaryFiler/age`, null)).get).toBe(65);
    expect(factGraph.get(Path.concretePath(`/secondaryFilerAgeDayOfDeath`, null)).get).toBe(65);
    expect(factGraph.get(Path.concretePath(`/secondaryFiler/age64OrYoungerForEitc`, null)).get).toBe(false);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
      [`/secondaryFilerDateOfDeath`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR)}-06-01`),
      [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 65}-06-02`),
    });
    expect(factGraph2.get(Path.concretePath(`/secondaryFiler/age`, null)).get).toBe(65);
    expect(factGraph2.get(Path.concretePath(`/secondaryFilerAgeDayOfDeath`, null)).get).toBe(64);
    expect(factGraph2.get(Path.concretePath(`/secondaryFiler/age64OrYoungerForEitc`, null)).get).toBe(true);
  });

  it(`A spouse who died the day before their 25th birthday is eligible for EITC`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/age64OrYoungerForEitc`];
    // If they die the day before their birthday, they still count as 25
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
      [`/secondaryFilerDateOfDeath`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR)}-06-01`),
      [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 25}-06-02`),
    });
    expect(factGraph.get(Path.concretePath(`/secondaryFiler/age`, null)).get).toBe(25);
    expect(factGraph.get(Path.concretePath(`/secondaryFilerAgeDayBeforeDeath`, null)).get).toBe(25);
    expect(factGraph.get(Path.concretePath(`/secondaryFiler/age25OrOlderForEitc`, null)).get).toBe(true);

    // If they die two days day before their birthday, they do not count as 25 for EITC
    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      '/maritalStatus': createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
      [`/secondaryFilerDateOfDeath`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR)}-06-01`),
      [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 25}-06-03`),
    });
    expect(factGraph2.get(Path.concretePath(`/secondaryFiler/age`, null)).get).toBe(25);
    expect(factGraph2.get(Path.concretePath(`/secondaryFilerAgeDayBeforeDeath`, null)).get).toBe(24);
    expect(factGraph2.get(Path.concretePath(`/secondaryFiler/age25OrOlderForEitc`, null)).get).toBe(false);
  });
});

describe(`Standard deduction age tests`, () => {
  it(`Someone who turns 65 on January 1 of the next tax year counts as 65`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/age65OrOlder`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 64}-01-01`),
    });
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age`, null)).get).toBe(64);
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age65OrOlder`, null)).get).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 64}-01-02`),
    });
    expect(factGraph2.get(Path.concretePath(`/primaryFiler/age`, null)).get).toBe(64);
    expect(factGraph2.get(Path.concretePath(`/primaryFiler/age65OrOlder`, null)).get).toBe(false);
  });
  it(`A 61 year old is below 65`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/age65OrOlder`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 61}-06-01`),
    });
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age`, null)).get).toBe(61);
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age65OrOlder`, null)).get).toBe(false);
  });

  it(`A 67 year old is above 65`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/age65OrOlder`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 67}-06-01`),
    });
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age`, null)).get).toBe(67);
    expect(factGraph.get(Path.concretePath(`/primaryFiler/age65OrOlder`, null)).get).toBe(true);
  });

  it(`MFJDeceasedSpouseReachedAge65ForStandardD calculates if a person turns 65 before they died`, ({ task }) => {
    task.meta.testedFactPaths = [`/MFJDeceasedSpouseReachedAge65ForStandardD`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 65}-01-02`),
      [`/secondaryFilerDateOfDeath`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR)}-01-03`),
    });

    expect(factGraph.get(Path.concretePath(`/secondaryFilerAgeDayBeforeDeath`, null)).get).toBe(65);
    expect(factGraph.get(Path.concretePath(`/MFJDeceasedSpouseReachedAge65ForStandardD`, null)).get).toBe(true);
  });

  it(`MFJDeceasedSpouseReachedAge65ForStandardD: 64 is younger than 65`, ({ task }) => {
    task.meta.testedFactPaths = [`/MFJDeceasedSpouseReachedAge65ForStandardD`];

    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR) - 64}-01-02`),
      [`/secondaryFilerDateOfDeath`]: createDayWrapper(`${Number.parseInt(CURRENT_TAX_YEAR)}-01-03`),
    });

    expect(factGraph.get(Path.concretePath(`/secondaryFilerAgeDayBeforeDeath`, null)).get).toBe(64);
    expect(factGraph.get(Path.concretePath(`/MFJDeceasedSpouseReachedAge65ForStandardD`, null)).get).toBe(false);
  });
});

describe(`Residency tests`, () => {
  it(`A filer claiming NY residency is identified as such in the fact graph`, ({ task }) => {
    task.meta.testedFactPaths = [`/livedInNy`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ny`, `/scopedStateOptions`),
    });
    expect(factGraph.get(Path.concretePath(`/livedInNy`, null)).get).toBe(true);
  });

  it(`A filer not in NY isn't identified as such in the fact graph`, ({ task }) => {
    task.meta.testedFactPaths = [`/livedInNy`];
    const { factGraph } = setupFactGraph({
      ...baseFilersData,
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ma`, `/scopedStateOptions`),
    });
    expect(factGraph.get(Path.concretePath(`/livedInNy`, null)).get).toBe(false);
  });
});
