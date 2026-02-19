import { describe, it, expect } from 'vitest';
import {
  createStringWrapper,
  createBooleanWrapper,
  createEnumWrapper,
  createCollectionWrapper,
  createDayWrapper,
  createTinWrapper,
  createDollarWrapper,
} from '../persistenceWrappers.js';
import { ConcretePath } from '@irs/js-factgraph-scala';
import { baseFilerData } from '../testData.js';
import { Path } from '../../flow/Path.js';
import { setupFactGraph } from '../setupFactGraph.js';

export const primaryFilerId = `959c03d1-af4a-447f-96aa-d19397048a44`;
export const spouseId = `859c03d1-af4a-447f-96aa-d19397048a48`;
export const uuid = `759c03d1-af4a-447f-96aa-d19397048a44`;

describe(`Dependency tests for a non-MFJ filer`, () => {
  const singleFilerData = {
    [`/filers/#${primaryFilerId}/firstName`]: createStringWrapper(`Test`),
    '/filers': createCollectionWrapper([primaryFilerId, spouseId]),
    [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`1987-01-01`),
    [`/filers/#${primaryFilerId}/isBlind`]: createBooleanWrapper(false),
    [`/filers/#${primaryFilerId}/isPrimaryFiler`]: createBooleanWrapper(true),
    [`/filers/#${spouseId}/isPrimaryFiler`]: createBooleanWrapper(false),
    '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
    '/filingStatus': createEnumWrapper(`single`, `/filingStatusOptions`),

    // Must make a US citizen, give them a TIN, and give them earned income to pass EITC eligibility
    [`/filers/#${primaryFilerId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
    '/formW2s': createCollectionWrapper([uuid]),
    [`/formW2s/#${uuid}/writableWages`]: createDollarWrapper(`1`),
  };

  it(`Eligible to claim dependents when the filer cannot be claimed`, ({ task }) => {
    task.meta.testedFactPaths = [`/treatFilersAsDependents`];

    const { factGraph } = setupFactGraph({
      ...singleFilerData,
      ...{ [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false) },
    });
    expect(factGraph.get(`/treatFilersAsDependents` as ConcretePath).get).toBe(false);
  });

  it(`When the filer can be claimed, but when claimer isn't filing or is filing for refund only,
      the filer can still claim dependents`, ({ task }) => {
    task.meta.testedFactPaths = [`/treatFilersAsDependents`];

    // When the claimer is filing for refund only
    const { factGraph: filingFactGraph } = setupFactGraph({
      ...singleFilerData,
      ...{
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/potentialClaimerDidFile`]: createBooleanWrapper(true),
        [`/primaryFilerPotentialClaimerFiledOnlyForRefund`]: createBooleanWrapper(true),
      },
    });
    expect(filingFactGraph.get(`/treatFilersAsDependents` as ConcretePath).get).toBe(false);

    // The claimer isn't filing
    const { factGraph } = setupFactGraph({
      ...singleFilerData,
      ...{
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/potentialClaimerDidFile`]: createBooleanWrapper(false),
      },
    });
    expect(factGraph.get(`/treatFilersAsDependents` as ConcretePath).get).toBe(false);
  });

  it(`When the filer can be claimed, and the claimer must file or is filing for refund and credits,
      the filer can no longer claim dependents `, ({ task }) => {
    // When the claimer must file
    task.meta.testedFactPaths = [`/treatFilersAsDependents`];
    const { factGraph: mustFileFactGraph } = setupFactGraph({
      ...singleFilerData,
      ...{
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
      },
    });
    expect(mustFileFactGraph.get(`/treatFilersAsDependents` as ConcretePath).get).toBe(true);
    task.meta.testedFactPaths = [`/treatFilersAsDependents`];

    // The claimer is filing for refund + credits
    const { factGraph } = setupFactGraph({
      ...singleFilerData,
      ...{
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/potentialClaimerDidFile`]: createBooleanWrapper(true),
        [`/primaryFilerPotentialClaimerFiledOnlyForRefund`]: createBooleanWrapper(false),
      },
    });
    expect(factGraph.get(`/treatFilersAsDependents` as ConcretePath).get).toBe(true);
  });

  it(`/treatFilersAsDependents is  still complete with MFS status`, ({ task }) => {
    task.meta.testedFactPaths = [`/treatFilersAsDependents`];
    const { factGraph } = setupFactGraph({
      ...singleFilerData,
      ...{
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatus': createEnumWrapper(`marriedFilingSeparately`, `/filingStatusOptions`),
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
        [`/filers/#${primaryFilerId}/potentialClaimerDidFile`]: createBooleanWrapper(true),
        [`/primaryFilerPotentialClaimerFiledOnlyForRefund`]: createBooleanWrapper(false),
      },
    });
    expect(factGraph.get(`/treatFilersAsDependents` as ConcretePath).get).toBe(true);

    const { factGraph: factGraph2 } = setupFactGraph({
      ...singleFilerData,
      ...{
        '/maritalStatus': createEnumWrapper(`married`, `/maritalStatusOptions`),
        '/filingStatus': createEnumWrapper(`marriedFilingSeparately`, `/filingStatusOptions`),
        [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
      },
    });
    expect(factGraph2.get(`/treatFilersAsDependents` as ConcretePath).get).toBe(false);
  });
});

describe(`validates the primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits fact`, () => {
  it(`potential dependent if required to file`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`];
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
    });
    expect(
      factGraph.get(
        Path.concretePath(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`, null)
      ).get
    ).toBe(true);
  });

  it(`potential dependent if not required to file and files for a non refund only tax return`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`];

    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/potentialClaimerDidFile`]: createBooleanWrapper(true),
      [`/primaryFilerPotentialClaimerFiledOnlyForRefund`]: createBooleanWrapper(false),
    });
    expect(
      factGraph.get(
        Path.concretePath(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`, null)
      ).get
    ).toBe(true);
  });

  it(`not a potential dependent if no one can claim them`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`];

    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
    });
    expect(
      factGraph.get(
        Path.concretePath(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`, null)
      ).get
    ).toBe(false);
  });

  it(`not a potential dependent if someone can claim them but they don't file`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`];

    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/potentialClaimerDidFile`]: createBooleanWrapper(false),
    });
    expect(
      factGraph.get(
        Path.concretePath(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`, null)
      ).get
    ).toBe(false);
  });

  it(`not a potential dependent if they can be claimed and they are filing a refund only return`, ({ task }) => {
    task.meta.testedFactPaths = [`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`];

    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/potentialClaimerDidFile`]: createBooleanWrapper(true),
      [`/primaryFilerPotentialClaimerFiledOnlyForRefund`]: createBooleanWrapper(true),
    });
    expect(
      factGraph.get(
        Path.concretePath(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`, null)
      ).get
    ).toBe(false);
  });
});

describe(`When MFJ`, () => {
  // These still have to be written
  it.todo(`Only modifies standard deduction when the user has decided to claim /MFJClaimingRefundOnly`);
  it.todo(`Only bans the dependency section when the user has decided to claim /MFJClaimingRefundOnly`);
  it.todo(`Only bans from EITC when the user has decided to claim /MFJClaimingRefundOnly`);
});
