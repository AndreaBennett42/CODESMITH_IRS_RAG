import { describe } from 'vitest';
import {
  FilingStatus,
  CollectionSubPath,
  baseFilerData,
  makeChildData,
  makeFilerData,
  makeW2Data,
  makeCareProviderData,
  primaryFilerId,
  spouseId,
  uuid,
  makeCollectionItem,
} from '../testData.js';
import { CURRENT_TAX_YEAR } from '../../constants/taxConstants.js';
import crypto from 'node:crypto';
import {
  createAddressWrapper,
  createBooleanWrapper,
  createCollectionItemWrapper,
  createCollectionWrapper,
  createDayWrapper,
  createDollarWrapper,
  createEnumWrapper,
  createStringWrapper,
  createTinWrapper,
  PersistenceWrapper,
} from '../persistenceWrappers.js';
import { Path } from '../../flow/Path.js';
import {
  AddressFactory,
  CollectionItemReferenceFactory,
  ConcretePath,
  DollarFactory,
  EinFactory,
  EnumFactory,
  FactGraph,
  jsSetToScalaSet,
  MultiEnumFactory,
  TinFactory,
} from '@irs/js-factgraph-scala';
import { setupFactGraph } from '../setupFactGraph.js';
import { CdccDisqualifyingItems } from '../../flow/flow-chunks/credits-and-deductions/CdccSubSubcategory.js';
import { Condition } from '../../flow/Condition.js';
import { ItemConfig } from '../../components/ConditionalList/ConditionalList.js';
import { AbsolutePath } from '../../fact-dictionary/Path.js';

const CURRENT_TAX_YEAR_AS_NUMBER = Number.parseInt(CURRENT_TAX_YEAR);

const CDCC_SUPPORTED_FILING_STATUSES = [
  FilingStatus.MFJ,
  FilingStatus.SINGLE,
  FilingStatus.MFS,
  // FilingStatus.HOH,
  // FilingStatus.QSS,
] as const;
const MARRIED_FILING_STATUS = [FilingStatus.MFJ, FilingStatus.MFS];

const w2Id = `9ba9d216-81a8-4944-81ac-9410b2fad150`;

const sharedData = {
  ...baseFilerData,
  [Path.concretePath(`/filers/*/canBeClaimed`, uuid)]: createBooleanWrapper(false),
};

const livedApartFacts = {
  [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
  [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
  [`/spouseLivedTogetherMonths`]: createEnumWrapper(
    `livedTogetherSixMonthsOrLess`,
    `/spouseLivedTogetherMonthsOptions`
  ),
};

const livedTogetherFacts = {
  [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
};

function makeClaimedChild(
  dependentId: string,
  childFacts: Partial<Record<CollectionSubPath<`/familyAndHousehold`>, PersistenceWrapper>>
) {
  return {
    ...makeChildData(dependentId),
    ...makeCollectionItem(dependentId, `/familyAndHousehold`, {
      ...childFacts,
      [`/tpClaims`]: createBooleanWrapper(true),
    }),
  };
}

const filterConditionalListItems = (factGraph: FactGraph, items: ItemConfig[], collectionId: string | null) => {
  return items.filter((item) => {
    if (item.conditions && item.conditions.length > 0) {
      // This doesn't use the collectionId because these conditional list items are not in a collection
      const result = item.conditions.every((c) => new Condition(c).evaluate(factGraph, collectionId));
      return result;
    }
    return true;
  });
};

describe(`For CDCC, a filer`, () => {
  describe.each(CDCC_SUPPORTED_FILING_STATUSES)(`with filing status %s`, (filingStatus) => {
    const isMarried = MARRIED_FILING_STATUS.includes(filingStatus);

    const w2Id2 = `4a939366-ee7c-4717-9794-5019a93b29dc`;
    const dependentId = crypto.randomUUID();

    const spouseData = makeFilerData(spouseId, { [`/canBeClaimed`]: createBooleanWrapper(false) });
    const baseData = {
      ...sharedData,
      ...(isMarried ? { ...livedApartFacts, ...spouseData } : {}),
      [`/maritalStatus`]: createEnumWrapper(isMarried ? `married` : `single`, `/maritalStatusOptions`),
      [`/filingStatus`]: createEnumWrapper(filingStatus, `/filingStatusOptions`),
      [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(true), // for MFS, needed for standard deduction
    };

    const eligibilityData = {
      [FilingStatus.MFJ]: [
        {
          // Both Filers have earned income
          description: `and both filers have earned income`,
          eligibilityDatum: {
            ...baseData,
            ...livedTogetherFacts,
            ...makeW2Data(50_000, w2Id),
            ...makeW2Data(5000, w2Id2, spouseId),
            '/formW2s': createCollectionWrapper([w2Id, w2Id2]),
          },
        },
        {
          description: `with a PF W2 and SF as a student`,
          eligibilityDatum: {
            ...baseData,
            ...livedTogetherFacts,
            ...makeW2Data(50_000, w2Id, primaryFilerId),
            [Path.concretePath(`/filers/*/isStudent`, spouseId)]: createBooleanWrapper(true),
            '/formW2s': createCollectionWrapper([w2Id]),
          },
        },
        {
          description: `with a SF W2 and PF has combat pay`,
          eligibilityDatum: {
            ...baseData,
            ...livedTogetherFacts,
            ...makeW2Data(50_000, w2Id2, spouseId),
            ...makeW2Data(0, w2Id),
            '/formW2s': createCollectionWrapper([w2Id, w2Id2]),
            [`/formW2s/#${w2Id}/writableCombatPay`]: createDollarWrapper(`1000.00`),
          },
        },
        {
          description: `while being MFJ dependents with a non-dependent QP`,
          eligibilityDatum: {
            ...baseData,
            ...livedTogetherFacts,

            ...makeW2Data(50_000, w2Id),
            ...makeW2Data(5000, w2Id2, spouseId),
            '/formW2s': createCollectionWrapper([w2Id, w2Id2]),

            ...makeChildData(dependentId, `${CURRENT_TAX_YEAR_AS_NUMBER - 8}-01-01`),
            [`/familyAndHousehold`]: createCollectionWrapper([dependentId]),
            [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(false),
            // is MFJDependent,
            [Path.concretePath(`/MFJRequiredToFile`, null)]: createBooleanWrapper(false),
            [Path.concretePath(`/MFJDependentsFilingForCredits`, null)]: createBooleanWrapper(false),
          },
        },
      ],
      [FilingStatus.SINGLE]: [
        {
          description: `with earned income`,
          eligibilityDatum: {
            ...baseData,
            ...makeW2Data(20_000, w2Id),
            '/formW2s': createCollectionWrapper([w2Id]),
          },
        },
        {
          description: `with combat pay`,
          eligibilityDatum: {
            ...baseData,
            ...makeW2Data(0, w2Id),
            '/formW2s': createCollectionWrapper([w2Id]),
            [`/formW2s/#${w2Id}/writableCombatPay`]: createDollarWrapper(`1000.00`),
          },
        },
      ],
      [FilingStatus.MFS]: [
        {
          description: `with earned income`,
          eligibilityDatum: {
            ...baseData,
            ...makeW2Data(20_000, w2Id),
            '/formW2s': createCollectionWrapper([w2Id]),
          },
        },
        {
          description: `with combat pay`,
          eligibilityDatum: {
            ...baseData,
            ...makeW2Data(0, w2Id),
            '/formW2s': createCollectionWrapper([w2Id]),
            [`/formW2s/#${w2Id}/writableCombatPay`]: createDollarWrapper(`1000.00`),
          },
        },
      ],
    };

    const ineligibilityData = {
      [FilingStatus.MFJ]: [
        {
          description: `with only PF earned income`,
          ineligibilityDatum: {
            ...baseData,
            ...livedTogetherFacts,
            ...makeW2Data(5000, w2Id),
            '/formW2s': createCollectionWrapper([w2Id]),
          },
        },
        {
          description: `with only SF earned income`,
          ineligibilityDatum: {
            ...baseData,
            ...livedTogetherFacts,
            ...makeW2Data(5000, w2Id2, spouseId),
            '/formW2s': createCollectionWrapper([w2Id2]),
          },
        },
        {
          description: `with no earned income, combat pay, neither filers are students/disabled`,
          ineligibilityDatum: {
            ...baseData,
            ...livedTogetherFacts,
          },
        },
      ],
      [FilingStatus.SINGLE]: [
        {
          description: `with no earned income`,
          ineligibilityDatum: {
            ...baseData,
          },
        },
        {
          description: `while a student`,
          ineligibilityDatum: {
            ...baseData,
            [Path.concretePath(`/filers/*/isStudent`, primaryFilerId)]: createBooleanWrapper(true),
          },
        },
      ],
      [FilingStatus.MFS]: [
        {
          description: `with no earned income`,
          ineligibilityDatum: {
            ...baseData,
          },
        },
        {
          description: `while a student`,
          ineligibilityDatum: {
            ...baseData,
            [Path.concretePath(`/filers/*/isStudent`, primaryFilerId)]: createBooleanWrapper(true),
          },
        },
      ],
    };

    describe.each(ineligibilityData[filingStatus])(`$description`, ({ ineligibilityDatum }) => {
      it(`could not qualify for CDCC`, () => {
        const { factGraph } = setupFactGraph({
          ...ineligibilityDatum,
        });

        const fact = factGraph.get(`/maybeEligibleForCdccBase` as ConcretePath);
        expect(fact.complete).toBe(true);
        expect(fact.get).toBe(false);
      });

      describe(`without dependent care benefits`, () => {
        it(`should not expected to file Form 2441`, () => {
          const { factGraph } = setupFactGraph({
            ...ineligibilityDatum,
          });

          const fact = factGraph.get(Path.concretePath(`/shouldSubmitForm2441`, null));
          expect(fact.complete).toBe(true);
          expect(fact.get).toBe(false);
        });
      });

      describe(`who carried forward benefits from last year`, () => {
        const carriedForwardData = {
          [Path.concretePath(`/hasCdccCarryoverAmountFromPriorTaxYear`, null)]: createBooleanWrapper(true),
          [Path.concretePath(`/writableCdccCarryoverAmountFromPriorTaxYear`, null)]: createDollarWrapper(`2000`),
        };

        it(`should be expected to file Form 2441`, () => {
          const { factGraph } = setupFactGraph({
            ...ineligibilityDatum,
            ...carriedForwardData,
          });

          const fact = factGraph.get(Path.concretePath(`/shouldSubmitForm2441`, null));
          expect(fact.complete).toBe(true);
          expect(fact.get).toBe(true);
        });
      });
    });

    describe.each(eligibilityData[filingStatus])(`$description`, ({ eligibilityDatum }) => {
      it(`could qualify for CDCC`, () => {
        const { factGraph } = setupFactGraph({
          ...eligibilityDatum,
        });

        const fact = factGraph.get(`/maybeEligibleForCdccBase` as ConcretePath);
        expect(fact.complete).toBe(true);
        expect(fact.get).toBe(true);
      });
    });

    describe(`with dependent care benefits`, () => {
      const benefitsAmount = 2000;
      const formW2WithBenefits = {
        ...makeW2Data(50_000, w2Id, primaryFilerId),
        [Path.concretePath(`/formW2s/*/writableDependentCareBenefits`, w2Id)]: createDollarWrapper(
          benefitsAmount.toString()
        ),
      };

      describe.each([
        { description: `fully utilized`, qualifyingExpenses: benefitsAmount },
        { description: `partially utilized`, qualifyingExpenses: benefitsAmount / 2 },
        { description: `unused`, qualifyingExpenses: 0 },
      ])(`$description`, ({ qualifyingExpenses }) => {
        it(`should be expected to file Form 2441`, () => {
          const { factGraph } = setupFactGraph({
            ...sharedData,
            ...formW2WithBenefits,
            [Path.concretePath(`/writableCdccTotalQualifiedDependentCareExpenses`, null)]: createDollarWrapper(
              qualifyingExpenses.toString()
            ),
          });

          const fact = factGraph.get(`/shouldSubmitForm2441` as ConcretePath);
          expect(fact.complete).toBe(true);
          expect(fact.get).toBe(true);
        });
      });
    });

    describe(`with a qualifying dependent`, () => {
      const qualifyingDependent = makeClaimedChild(dependentId, {
        [`/unableToCareForSelf`]: createBooleanWrapper(true),
        [`/residencyDuration`]: createEnumWrapper(`allYear`, `residencyDurationOptions`),
        [`/tpPaidMostOfHomeUpkeep`]: createBooleanWrapper(true),
      });

      describe.each([
        { description: `and some qualifying expenses not provided by employer`, qualifyingExpenses: 2000 },
        { description: `and no qualifying expenses` },
      ])(`$description`, ({ qualifyingExpenses }) => {
        const hasQualifyingExpenses = qualifyingExpenses !== undefined;

        it(`${hasQualifyingExpenses ? `should` : `should not`} be expected to file Form 2441`, () => {
          const { factGraph } = setupFactGraph({
            ...eligibilityData[filingStatus][0].eligibilityDatum,
            ...(filingStatus === FilingStatus.MFS
              ? {
                  '/MFSLivingSpouseFilingReturn': createBooleanWrapper(false),
                  '/MFSSpouseHasGrossIncome': createBooleanWrapper(false),
                  '/MFSSpouse65OrOlder': createBooleanWrapper(false),
                  ...livedApartFacts,
                }
              : {}),
            ...qualifyingDependent,
            [Path.concretePath(`/familyAndHousehold`, null)]: createCollectionWrapper([dependentId]),
            [Path.concretePath(`/writableCdccHasQualifyingExpenses`, null)]:
              createBooleanWrapper(hasQualifyingExpenses),
            [Path.concretePath(`/familyAndHousehold/*/cdccHasDependentCareExpenses`, dependentId)]:
              createBooleanWrapper(hasQualifyingExpenses),
            [Path.concretePath(`/familyAndHousehold/*/cdccHadExpensesPaidToQualifyingProvider`, dependentId)]:
              createBooleanWrapper(hasQualifyingExpenses),
            [Path.concretePath(`/familyAndHousehold/*/writableCdccQualifyingExpenseAmount`, dependentId)]:
              createDollarWrapper(qualifyingExpenses?.toString() ?? `0`),
          });

          const fact = factGraph.get(`/shouldSubmitForm2441` as ConcretePath);
          expect(fact.complete).toBe(true);
          expect(fact.get).toBe(hasQualifyingExpenses);
        });
      });
    });

    if (FilingStatus.MFS === filingStatus) {
      describe(`who is MFS and lived apart from their spouse during the last 6 months of the tax year`, () => {
        it(`is considered unmarried for purposes of claiming the credit on Form 2441`, () => {
          const { factGraph } = setupFactGraph({
            ...baseData,
            ...livedApartFacts,
          });

          const fact = factGraph.get(`/mfsFilerCouldBeConsiderUnmarriedForCdcc` as ConcretePath);

          expect(fact.complete).toBe(true);
          expect(fact.get).toBe(true);
        });
      });

      describe(`who did not live apart from their spouse during the last 6 months of the tax year`, () => {
        it(`is considered married for purposes of claiming the credit on Form 2441`, () => {
          const { factGraph } = setupFactGraph({
            ...baseData,
            ...livedTogetherFacts,
          });

          const fact = factGraph.get(`/mfsFilerCouldBeConsiderUnmarriedForCdcc` as ConcretePath);

          expect(fact.complete).toBe(true);
          expect(fact.get).toBe(false);
        });
      });
    }

    describe(`with no dependents or qualifying persons`, () => {
      const { factGraph } = setupFactGraph({
        ...baseData,
        ...makeW2Data(50000, w2Id),
        [Path.concretePath(`/formW2s`, null)]: createCollectionWrapper([w2Id]),
      });

      it.each([
        `/cdccHasQualifyingExpenses`,
        `/cdccQualified`,
        `/maybeEligibleForCdcc`,
        `/cdccHasQualifyingPersons`,
        `/shouldSubmitForm2441`,
      ] as ConcretePath[])(`%s is false`, (path) => {
        const fact = factGraph.get(path);

        expect(fact.complete).toBe(true);
        expect(fact.get).toBe(false);
      });

      it(`/cdccTotalCredit is zero`, () => {
        const fact = factGraph.get(`/cdccTotalCredit` as ConcretePath);
        expect(fact.complete).toBe(true);
        expect(fact.get.toString()).toBe(`0.00`);
      });
    });

    describe(`with no dependents or non-filer qualifying person`, () => {
      describe(`and Primary filer is disabled`, () => {
        const cohabitationScenarios = isMarried
          ? [
              {
                status: `together`,
                cohabitationFacts: livedTogetherFacts,
                shouldBeQualifyingPerson: filingStatus === FilingStatus.MFJ,
                // MFJ should qualify for the credit, but MFS should not
                shouldQualifyForCredit: filingStatus === FilingStatus.MFJ,
              },
              {
                status: `apart`,
                cohabitationFacts: livedApartFacts,
                // MFS is considered unmarried if living apart and cannot be a qualifying person
                shouldBeQualifyingPerson: false,
                shouldQualifyForCredit: false,
              },
            ]
          : [
              {
                status: `alone`,
                cohabitationFacts: {},
                shouldBeQualifyingPerson: false,
                shouldQualifyForCredit: false,
              },
            ];

        describe.each(cohabitationScenarios)(
          `living $status`,
          ({ status, cohabitationFacts, shouldBeQualifyingPerson, shouldQualifyForCredit }) => {
            const { factGraph } = setupFactGraph({
              ...baseData,
              ...makeW2Data(50000, w2Id, spouseId),
              [Path.concretePath(`/formW2s`, null)]: createCollectionWrapper([w2Id]),
              ...cohabitationFacts,
              [Path.concretePath(`/filers/*/isDisabled`, uuid)]: createBooleanWrapper(true),
            });

            it.each([`/cdccHasQualifyingFilers`, `/primaryFiler/cdccQualifyingPerson`] as ConcretePath[])(
              `%s is ${shouldBeQualifyingPerson}`,
              (path) => {
                const fact = factGraph.get(path);
                expect(fact.complete).toBe(true);
                expect(fact.get).toBe(shouldBeQualifyingPerson);
              }
            );

            const expectedMfsButEligibleForCdcc = filingStatus === FilingStatus.MFS && status === `apart`;
            it(`/mfsButEligibleForCdcc is ${expectedMfsButEligibleForCdcc}`, () => {
              const fact = factGraph.get(`/mfsButEligibleForCdcc` as ConcretePath);
              expect(fact.complete).toBe(true);
              expect(fact.get).toBe(false);
            });

            it.each([
              { path: `/maybeEligibleForCdcc`, expectedValue: shouldQualifyForCredit },
              { path: `/cdccHasQualifyingPersons`, expectedValue: shouldBeQualifyingPerson },
            ])(`$path is $expectedValue`, ({ path, expectedValue }) => {
              const fact = factGraph.get(path as ConcretePath);
              expect(fact.complete).toBe(true);
              expect(fact.get).toBe(expectedValue);
            });

            if (!shouldQualifyForCredit) {
              it(`/cdccTotalCredit is zero`, () => {
                const fact = factGraph.get(`/cdccTotalCredit` as ConcretePath);
                expect(fact.complete).toBe(true);
                expect(fact.get.toString()).toBe(`0.00`);
              });
            }
          }
        );
      });

      if (isMarried) {
        describe(`and Secondary filer is disabled`, () => {
          const cohabitationScenarios = [
            [`together`, livedTogetherFacts, filingStatus === FilingStatus.MFJ, filingStatus === FilingStatus.MFJ],
            [`apart`, livedApartFacts, false, false],
          ] as const;

          describe.each(cohabitationScenarios)(
            `living %s`,
            (status, cohabitationFacts, shouldBeQualifyingPerson, shouldQualifyForCredit) => {
              const { factGraph } = setupFactGraph({
                ...baseData,
                ...makeW2Data(50000, w2Id),
                [Path.concretePath(`/formW2s`, null)]: createCollectionWrapper([w2Id]),
                [Path.concretePath(`/formW2s/*/filer`, w2Id)]: createCollectionItemWrapper(primaryFilerId),
                ...cohabitationFacts,
                [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(true),
              });

              it.each([`/cdccHasQualifyingFilers`, `/secondaryFiler/cdccQualifyingPerson`])(
                `%s is ${shouldBeQualifyingPerson}`,
                (factPath) => {
                  const fact = factGraph.get(factPath as ConcretePath);
                  expect(fact.complete).toBe(true);
                  expect(fact.get).toBe(shouldBeQualifyingPerson);
                }
              );

              // MFS can never qualify for the CDCC credit with just their spouse as a qualifying person
              it(`/mfsButEligibleForCdcc is false`, () => {
                const fact = factGraph.get(`/mfsButEligibleForCdcc` as ConcretePath);
                expect(fact.complete).toBe(true);
                expect(fact.get).toBe(false);
              });

              it.each([
                { path: `/maybeEligibleForCdcc`, expectedValue: shouldQualifyForCredit },
                { path: `/cdccHasQualifyingPersons`, expectedValue: shouldBeQualifyingPerson },
              ])(`$path is $expectedValue`, ({ path, expectedValue }) => {
                const fact = factGraph.get(path as ConcretePath);
                expect(fact.complete).toBe(true);
                expect(fact.get).toBe(expectedValue);
              });

              if (!shouldQualifyForCredit) {
                it(`/cdccTotalCredit is zero`, () => {
                  const fact = factGraph.get(`/cdccTotalCredit` as ConcretePath);
                  expect(fact.complete).toBe(true);
                  expect(fact.get.toString()).toBe(`0.00`);
                });
              }
            }
          );
        });
      }
    });

    const _baseData = baseData;
    describe(`Worksheet A routing`, () => {
      const baseData = {
        ..._baseData,
        ...makeW2Data(50000, w2Id),
        [Path.concretePath(`/formW2s`, null)]: createCollectionWrapper([w2Id]),
      };

      it(`is a knockout when conditions for requiring Worksheet A are met`, () => {
        const { factGraph } = setupFactGraph({
          ...baseData,
          [Path.concretePath(`/cdccHasCreditForPriorYearExpenses`, null)]: createBooleanWrapper(true),
          [Path.concretePath(`/cdccClaimedCreditForPriorYearExpenses`, null)]: createBooleanWrapper(false),
          [Path.concretePath(`/cdccClaimedMaxCreditForPriorTaxYear`, null)]: createBooleanWrapper(true),
        });

        const cdccHasPriorYearExpensesAndDidNotClaimCreditOrMaxExpenses = factGraph.get(
          `/knockoutCdccWorksheetA` as ConcretePath
        );
        expect(cdccHasPriorYearExpensesAndDidNotClaimCreditOrMaxExpenses.complete).toBe(true);
        expect(cdccHasPriorYearExpensesAndDidNotClaimCreditOrMaxExpenses.get).toBe(true);
      });

      it(`is not a knockout when conditions for not requiring Worksheet A are met`, () => {
        const { factGraph } = setupFactGraph({
          ...baseData,
          [Path.concretePath(`/cdccHasCreditForPriorYearExpenses`, null)]: createBooleanWrapper(true),
          [Path.concretePath(`/cdccClaimedCreditForPriorYearExpenses`, null)]: createBooleanWrapper(true),
          [Path.concretePath(`/cdccClaimedMaxCreditForPriorTaxYear`, null)]: createBooleanWrapper(true),
        });

        const cdccHasPriorYearExpensesAndDidNotClaimCreditOrMaxExpenses = factGraph.get(
          `/knockoutCdccWorksheetA` as ConcretePath
        );
        expect(cdccHasPriorYearExpensesAndDidNotClaimCreditOrMaxExpenses.complete).toBe(true);
        expect(cdccHasPriorYearExpensesAndDidNotClaimCreditOrMaxExpenses.get).toBe(false);
      });
    });

    describe.each([
      // Qualifying children, 13 or younger at end of tax year
      { dob: `${CURRENT_TAX_YEAR_AS_NUMBER - 10}-01-01`, shouldQualify: true },
      { dob: `${CURRENT_TAX_YEAR_AS_NUMBER - 13}-01-02`, shouldQualify: true },
      { dob: `${CURRENT_TAX_YEAR_AS_NUMBER - 13}-01-01`, shouldQualify: false }, // must be under 13 Jan 1
      { dob: `${CURRENT_TAX_YEAR_AS_NUMBER - 13}-12-31`, shouldQualify: true },
      // Qualifying person, older than 13 but unable to care for self
      { dob: `${CURRENT_TAX_YEAR_AS_NUMBER - 14}-01-01`, shouldQualify: false, unableToCareForSelf: false },
      {
        dob: `${CURRENT_TAX_YEAR_AS_NUMBER - 14}-01-01`,
        shouldQualify: true,
        unableToCareForSelf: true,
        livedWithFiler: `sixToElevenMonths`,
      },
      { dob: `${CURRENT_TAX_YEAR_AS_NUMBER - 18}-01-01`, shouldQualify: false, unableToCareForSelf: false },
      {
        dob: `${CURRENT_TAX_YEAR_AS_NUMBER - 18}-01-01`,
        shouldQualify: true,
        unableToCareForSelf: true,
        livedWithFiler: `sixToElevenMonths`,
      },
      // TODO: Special case qualifying persons
    ])(
      `and potentially qualifying person with birthdate: $dob, unableToCareForSelf: $unableToCareForSelf, ` +
        `livedWithFiler: $livedWithFiler`,
      ({ dob, shouldQualify, unableToCareForSelf, livedWithFiler }) => {
        const dependentId = crypto.randomUUID();
        const data = {
          ...baseData,
          ...makeW2Data(50000, w2Id),
          ...(isMarried
            ? {
                ...makeW2Data(1000, w2Id2, spouseId),
                '/formW2s': createCollectionWrapper([w2Id, w2Id2]),
              }
            : { '/formW2s': createCollectionWrapper([w2Id]) }),
          ...makeClaimedChild(dependentId, {
            [`/tpPaidMostOfHomeUpkeep`]: createBooleanWrapper(true),
            [`/dateOfBirth`]: createDayWrapper(dob),
            [`/tpClaims`]: createBooleanWrapper(true),
          }),
          [`/familyAndHousehold`]: createCollectionWrapper([dependentId]),
        };

        const { factGraph } = setupFactGraph(data);

        if (undefined !== unableToCareForSelf) {
          factGraph.set(
            Path.concretePath(`/familyAndHousehold/*/unableToCareForSelf`, dependentId),
            unableToCareForSelf
          );
        }
        if (undefined !== livedWithFiler) {
          factGraph.set(
            Path.concretePath(`/familyAndHousehold/*/residencyDuration`, dependentId),
            EnumFactory(livedWithFiler, `/residencyDurationOptions`).right
          );
        }

        factGraph.save();

        it.each([`/cdccHasQualifyingPersons`, `/maybeEligibleForCdcc`] as ConcretePath[])(
          `%s is ${shouldQualify}`,
          (path) => {
            const fact = factGraph.get(path);
            expect(fact.complete).toBe(true);
            expect(fact.get).toBe(shouldQualify);
          }
        );

        it.each([
          [`/isClaimedPersonOrEligibleByBenefitSplit`, true],
          [`/residencyTest`, livedWithFiler !== `lessThanSixMonths`],
        ] as const)(`/familyAndHousehold/[dependentId]%s is %s`, (path, expectedValue) => {
          const fact = factGraph.get(Path.concretePath(`/familyAndHousehold/*${path}`, dependentId));

          if (expectedValue !== undefined) {
            expect(fact.complete).toBe(true);
            expect(fact.get).toBe(expectedValue);
          } else if (fact.complete) {
            expect(fact.get).toBe(false);
          }
        });
      }
    );

    describe.each([
      // IF your 2022 adjusted gross income was:	 	THEN the decimal amount is:
      // Over:	 	But not over:
      //  $ 0	    —	$15,000	 	0.35
      { agi: 0, decimalAmount: 0 },
      { agi: 1, decimalAmount: 0.35 },
      { agi: 10000, decimalAmount: 0.35 },
      { agi: 15000, decimalAmount: 0.35 },
      // 15,000	—	17,000	 	0.34
      { agi: 15001, decimalAmount: 0.34 },
      { agi: 16000, decimalAmount: 0.34 },
      { agi: 17000, decimalAmount: 0.34 },
      // 17,000	—	19,000	 	0.33
      { agi: 17001, decimalAmount: 0.33 },
      { agi: 18001, decimalAmount: 0.33 },
      { agi: 19000, decimalAmount: 0.33 },
      // 19,000	—	21,000	 	0.32
      { agi: 19001, decimalAmount: 0.32 },
      { agi: 20000, decimalAmount: 0.32 },
      { agi: 21000, decimalAmount: 0.32 },
      // 21,000	—	23,000	 	0.31
      { agi: 21001, decimalAmount: 0.31 },
      { agi: 22000, decimalAmount: 0.31 },
      { agi: 23000, decimalAmount: 0.31 },
      // 23,000	—	25,000	 	0.30
      { agi: 23001, decimalAmount: 0.3 },
      { agi: 24000, decimalAmount: 0.3 },
      { agi: 25000, decimalAmount: 0.3 },
      // 25,000	—	27,000	 	0.29
      { agi: 25001, decimalAmount: 0.29 },
      { agi: 26000, decimalAmount: 0.29 },
      { agi: 27000, decimalAmount: 0.29 },
      // 27,000	—	29,000	 	0.28
      { agi: 27001, decimalAmount: 0.28 },
      { agi: 28000, decimalAmount: 0.28 },
      { agi: 29000, decimalAmount: 0.28 },
      // 29,000	—	31,000	 	0.27
      { agi: 29001, decimalAmount: 0.27 },
      { agi: 30000, decimalAmount: 0.27 },
      { agi: 31000, decimalAmount: 0.27 },
      // 31,000	—	33,000	 	0.26
      { agi: 31001, decimalAmount: 0.26 },
      { agi: 32000, decimalAmount: 0.26 },
      { agi: 33000, decimalAmount: 0.26 },
      // 33,000	—	35,000	 	0.25
      { agi: 33001, decimalAmount: 0.25 },
      { agi: 34000, decimalAmount: 0.25 },
      { agi: 35000, decimalAmount: 0.25 },
      // 35,000	—	37,000	 	0.24
      { agi: 35001, decimalAmount: 0.24 },
      { agi: 36000, decimalAmount: 0.24 },
      { agi: 37000, decimalAmount: 0.24 },
      // 37,000	—	39,000	 	0.23
      { agi: 37001, decimalAmount: 0.23 },
      { agi: 38000, decimalAmount: 0.23 },
      { agi: 39000, decimalAmount: 0.23 },
      // 39,000	—	41,000	 	0.22
      { agi: 39001, decimalAmount: 0.22 },
      { agi: 40000, decimalAmount: 0.22 },
      { agi: 41000, decimalAmount: 0.22 },
      // 41,000	—	43,000	 	0.21
      { agi: 41001, decimalAmount: 0.21 },
      { agi: 42000, decimalAmount: 0.21 },
      { agi: 43000, decimalAmount: 0.21 },
      // 43,000	—	No limit	0.20
      { agi: 43001, decimalAmount: 0.2 },
      { agi: 44000, decimalAmount: 0.2 },
      { agi: 45000, decimalAmount: 0.2 },
      { agi: 50000, decimalAmount: 0.2 },
      { agi: 100000, decimalAmount: 0.2 },
      { agi: 150000, decimalAmount: 0.2 },
    ])(`with an agi of $agi`, ({ agi, decimalAmount }) => {
      it(`has a /cdccCareExpensesDecimalAmount of ${decimalAmount}`, () => {
        const { factGraph } = setupFactGraph({
          ...baseData,
          ...makeW2Data(agi, w2Id),
          [Path.concretePath(`/formW2s`, null)]: createCollectionWrapper([w2Id]),
        });

        expect(factGraph.get(Path.concretePath(`/agi`, null)).get.toString()).toBe(agi.toFixed(2));
        expect(factGraph.get(Path.concretePath(`/cdccCareExpensesDecimalAmount`, null)).get.toString()).toBe(
          decimalAmount.toFixed(2)
        );
      });
    });
  });
});

/**
 * This set of tests checks the dependentCareBenefitsIsComplete fact.
 * It is a boolean fact that is true if the dependent care benefits section is complete.
 */
describe(`For /dependentCareBenefitsIsComplete, a filer`, () => {
  const spouseData = makeFilerData(spouseId, { [`/canBeClaimed`]: createBooleanWrapper(false) });

  describe.each([
    {
      description: `with filing status ${FilingStatus.MFJ}`,
      filingStatus: FilingStatus.MFJ,
      additionalFacts: { ...livedApartFacts, ...spouseData },
    },
    { description: `with filing status ${FilingStatus.SINGLE}`, filingStatus: FilingStatus.SINGLE },
    {
      description: `with filing status ${FilingStatus.MFS} and considered unmarried`,
      filingStatus: FilingStatus.MFS,
      additionalFacts: livedApartFacts,
    },
  ])(``, ({ filingStatus, additionalFacts }) => {
    const baseData = {
      ...sharedData,
      ...makeW2Data(50000, w2Id),
      ...additionalFacts,
      [`/maritalStatus`]: createEnumWrapper(
        filingStatus === FilingStatus.MFJ ? `married` : `single`,
        `/maritalStatusOptions`
      ),
      [`/filingStatus`]: createEnumWrapper(filingStatus, `/filingStatusOptions`),
      [Path.concretePath(`/formW2s`, null)]: createCollectionWrapper([w2Id]),
      [`/flowKnockoutHouseholdEmployee`]: createBooleanWrapper(false),
    };
    const careProviderUuid = crypto.randomUUID();

    describe(`if no reported or carryover dependent care benefits`, () => {
      const { factGraph } = setupFactGraph({
        ...baseData,
        '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(false),
      });

      it(`dependent care benefits section is complete`, () => {
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/dependentCareBenefitsIsComplete`, null)).get.toString()).toBe(`true`);
      });
    });

    describe(`if the filer had a household employee`, () => {
      const { factGraph } = setupFactGraph({
        ...baseData,
        [`/flowKnockoutHouseholdEmployee`]: createBooleanWrapper(true),
      });

      it(`triggers the application knockout state`, () => {
        expect(factGraph.get(Path.concretePath(`/flowIsKnockedOut`, null)).complete.toString()).toBe(`true`);
      });
    });

    describe(`if reported dependent care benefits but has no qualifying persons`, () => {
      const { factGraph } = setupFactGraph({
        ...baseData,
        ...makeW2Data(50000, w2Id),
        '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(true),
        '/writableCdccCarryoverAmountFromPriorTaxYear': createDollarWrapper(`1000`),
        '/writablePrimaryFilerDependentCarePlanMaximum': createDollarWrapper(`2500`),
        [`/formW2s`]: createCollectionWrapper([w2Id]),
      });

      it(`dependent care benefits section is complete`, () => {
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/dependentCareBenefitsIsComplete`, null)).get.toString()).toBe(`true`);
      });
    });

    describe(`if carryover dependent care benefits reported`, () => {
      const { factGraph } = setupFactGraph({
        ...baseData,
        '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(true),
      });

      it(`dependent care benefits section is incomplete`, () => {
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`false`);
        expect(factGraph.get(Path.concretePath(`/dependentCareBenefitsIsComplete`, null)).get.toString()).toBe(`false`);
      });
    });

    describe(`if dependent care benefits reported in box10`, () => {
      const dependentId = `85501abd-9f7c-4746-b4e9-f765565ca923`;

      const qualifyingDependent = {
        ...makeChildData(dependentId, `${CURRENT_TAX_YEAR_AS_NUMBER - 8}-01-01`),
        [`/familyAndHousehold`]: createCollectionWrapper([dependentId]),
        [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(false),
      };
      const { factGraph } = setupFactGraph({
        ...baseData,
        ...qualifyingDependent,
        '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(false),
        [`/formW2s`]: createCollectionWrapper([w2Id]),
        [`/formW2s/#${w2Id}/writableDependentCareBenefits`]: createDollarWrapper(`1000`),
        [`/formW2s/#${w2Id}/filer`]: createCollectionItemWrapper(primaryFilerId),
      });

      it(`dependent care benefits section is incomplete`, () => {
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`false`);
        expect(factGraph.get(Path.concretePath(`/dependentCareBenefitsIsComplete`, null)).get.toString()).toBe(`false`);
      });
    });

    describe(`if dependent care benefits reported in box10 and section is filled in`, () => {
      const data = {
        ...baseData,
        '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(false),
        [`/formW2s`]: createCollectionWrapper([w2Id]),
        [`/formW2s/#${w2Id}/writableDependentCareBenefits`]: createDollarWrapper(`1000`),
        [`/formW2s/#${w2Id}/filer`]: createCollectionItemWrapper(primaryFilerId),
        '/writableHasCdccForfeitedCredits': createBooleanWrapper(false),
        '/writableCdccTotalQualifiedDependentCareExpenses': createDollarWrapper(`4000`),
        '/writablePrimaryFilerDependentCarePlanMaximum': createDollarWrapper(`2500`),
        '/cdccCareProvidersIsDone': createBooleanWrapper(true),
        ...makeCareProviderData(careProviderUuid),
        '/cdccCareProviders': createCollectionWrapper([careProviderUuid]),
      };
      const { factGraph } = setupFactGraph(data);

      it(`dependent care benefits section is complete`, () => {
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/dependentCareBenefitsIsComplete`, null)).get.toString()).toBe(`true`);
      });
    });

    describe(`if only carryover dependent care benefits reported and section is filled in`, () => {
      const { factGraph } = setupFactGraph({
        ...baseData,
        '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(true),
        '/writableCdccTotalQualifiedDependentCareExpenses': createDollarWrapper(`4000`),
        '/writablePrimaryFilerDependentCarePlanMaximum': createDollarWrapper(`2500`),
        '/writableCdccCarryoverAmountFromPriorTaxYear': createDollarWrapper(`240`),
        '/cdccCareProvidersIsDone': createBooleanWrapper(true),
        ...makeCareProviderData(careProviderUuid),
        '/cdccCareProviders': createCollectionWrapper([careProviderUuid]),
      });

      it(`dependent care benefits section is complete`, () => {
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/dependentCareBenefitsIsComplete`, null)).get.toString()).toBe(`true`);
      });
    });
  });

  describe(`with filing status MFJ, special earned income`, () => {
    const w2Id2 = `39483641-eef1-4c55-8c9f-f1ae144b5c8b`;
    const dependentId = `85501abd-9f7c-4746-b4e9-f765565ca923`;
    const careProviderUuid = crypto.randomUUID();

    const qualifyingDependent = {
      ...makeChildData(dependentId, `${CURRENT_TAX_YEAR_AS_NUMBER - 8}-01-01`),
      [`/familyAndHousehold`]: createCollectionWrapper([dependentId]),
      [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(false),
    };
    const baseData = {
      ...sharedData,
      ...makeFilerData(primaryFilerId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
      [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/filingStatus`]: createEnumWrapper(FilingStatus.MFJ, `/filingStatusOptions`),
      ...qualifyingDependent,
      ...makeW2Data(50000, w2Id),
      ...makeW2Data(2000, w2Id2),
      [Path.concretePath(`/formW2s`, null)]: createCollectionWrapper([w2Id, w2Id2]),
      [Path.concretePath(`/formW2s/*/filer`, w2Id)]: createCollectionItemWrapper(primaryFilerId),
      [Path.concretePath(`/formW2s/*/filer`, w2Id2)]: createCollectionItemWrapper(spouseId),
      [Path.concretePath(`/filers/*/isStudent`, primaryFilerId)]: createBooleanWrapper(false),
      [Path.concretePath(`/filers/*/isStudent`, spouseId)]: createBooleanWrapper(false),
      [Path.concretePath(`/livedTogetherAllYearWithSpouse`, null)]: createBooleanWrapper(true),
      '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(true),
      '/writableHasCdccForfeitedCredits': createBooleanWrapper(false),
      '/writableCdccTotalQualifiedDependentCareExpenses': createDollarWrapper(`4000`),
      '/writablePrimaryFilerDependentCarePlanMaximum': createDollarWrapper(`2500`),
      '/writableCdccCarryoverAmountFromPriorTaxYear': createDollarWrapper(`240`),
      '/hasMfjSpouseEarnedIncome': createBooleanWrapper(true),
      '/writableMfsSpouseEarnedIncome': createDollarWrapper(`25000`),
      [Path.concretePath(`/cdccCareProvidersIsDone`, null)]: createBooleanWrapper(true),
      ...makeCareProviderData(careProviderUuid),
      '/cdccCareProviders': createCollectionWrapper([careProviderUuid]),
      [`/flowKnockoutHouseholdEmployee`]: createBooleanWrapper(false),
    };

    describe(`if only tp is disabled and doesn't answer < 250 question`, () => {
      const data = {
        ...baseData,
        [Path.concretePath(`/filers/*/isDisabled`, primaryFilerId)]: createBooleanWrapper(true),
        [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(false),
      };
      const { factGraph } = setupFactGraph(data);
      expect(factGraph.getValue(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)).toBe(`true`);
      it(`dependent care benefits section is incomplete`, () => {
        expect(factGraph.getValue(`/dependentCareBenefitsIsComplete`, null)).toBe(`false`);
      });
    });

    describe(`if only tp is disabled and did not make < 250`, () => {
      const data = {
        ...baseData,
        [Path.concretePath(`/filers/*/isDisabled`, primaryFilerId)]: createBooleanWrapper(true),
        [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(false),
        [Path.concretePath(`/primaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
          createBooleanWrapper(false),
      };
      const { factGraph } = setupFactGraph(data);
      expect(factGraph.getValue(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)).toBe(`true`);
      it(`dependent care benefits section is complete`, () => {
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.getValue(`/dependentCareBenefitsIsComplete`, null)).toBe(`true`);
      });
    });
    describe(`if only spouse is disabled and didn't answer < 250 question`, () => {
      const data = {
        ...baseData,
        [Path.concretePath(`/filers/*/isDisabled`, primaryFilerId)]: createBooleanWrapper(false),
        [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(true),
      };
      const { factGraph } = setupFactGraph(data);
      it(`dependent care benefits section is incomplete`, () => {
        expect(factGraph.getValue(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)).toBe(`true`);
        expect(factGraph.getValue(`/dependentCareBenefitsIsComplete`, null)).toBe(`false`);
      });
    });
    describe(`if only spouse is disabled and did not make < 250`, () => {
      const data = {
        ...baseData,
        [Path.concretePath(`/filers/*/isDisabled`, primaryFilerId)]: createBooleanWrapper(false),
        [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(true),
        [Path.concretePath(`/secondaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
          createBooleanWrapper(false),
      };
      const { factGraph } = setupFactGraph(data);
      it(`dependent care benefits section is complete`, () => {
        expect(factGraph.getValue(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.getValue(`/dependentCareBenefitsIsComplete`, null)).toBe(`true`);
      });
    });
    describe(`if both are disabled and didn't answer < 250 question`, () => {
      const data = {
        ...baseData,
        [Path.concretePath(`/filers/*/isDisabled`, primaryFilerId)]: createBooleanWrapper(true),
        [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(true),
      };
      const { factGraph } = setupFactGraph(data);
      it(`dependent care benefits section is complete`, () => {
        expect(factGraph.getValue(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.getValue(`/dependentCareBenefitsIsComplete`, null)).toBe(`false`);
      });
    });
    describe(`if both are disabled and only tp has answered < 250 question`, () => {
      const data = {
        ...baseData,
        [Path.concretePath(`/filers/*/isDisabled`, primaryFilerId)]: createBooleanWrapper(true),
        [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(true),
        [Path.concretePath(`/primaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
          createBooleanWrapper(false),
      };
      const { factGraph } = setupFactGraph(data);
      it(`dependent care benefits section is incomplete`, () => {
        expect(factGraph.getValue(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)).toBe(`true`);
        expect(factGraph.getValue(`/dependentCareBenefitsIsComplete`, null)).toBe(`false`);
      });
    });
    describe(`if both are disabled and both answered < 250 question`, () => {
      const data = {
        ...baseData,
        [Path.concretePath(`/filers/*/isDisabled`, primaryFilerId)]: createBooleanWrapper(true),
        [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(true),
        [Path.concretePath(`/primaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
          createBooleanWrapper(false),
        [Path.concretePath(`/secondaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
          createBooleanWrapper(false),
      };
      const { factGraph } = setupFactGraph(data);
      it(`dependent care benefits section is true`, () => {
        expect(factGraph.getValue(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.getValue(`/dependentCareBenefitsIsComplete`, null)).toBe(`true`);
      });
    });

    describe(`if Filer made < 250`, () => {
      it(`and filer is disabled, triggers the special earned income knockout`, () => {
        const datum = {
          ...baseData,
          [Path.concretePath(`/filers/*/isDisabled`, primaryFilerId)]: createBooleanWrapper(true),
          [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(false),
          [Path.concretePath(`/primaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
            createBooleanWrapper(true),
        };
        const { factGraph } = setupFactGraph(datum);
        expect(factGraph.getValue(`/knockoutStudentOrDisabled`, null)).toBe(`true`);
      });

      it(`and filer is a student, triggers the special earned income knockout`, () => {
        const datum = {
          ...baseData,
          [Path.concretePath(`/filers/*/isStudent`, primaryFilerId)]: createBooleanWrapper(true),
          [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(false),
          [Path.concretePath(`/primaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
            createBooleanWrapper(true),
        };
        const { factGraph } = setupFactGraph(datum);
        expect(factGraph.getValue(`/knockoutStudentOrDisabled`, null)).toBe(`true`);
      });
    });
    describe(`if Spouse is made < 250`, () => {
      it(`and spouse is disabled, triggers the special earned income knockout`, () => {
        const datum = {
          ...baseData,
          [Path.concretePath(`/filers/*/isDisabled`, primaryFilerId)]: createBooleanWrapper(false),
          [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(true),
          [Path.concretePath(`/secondaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
            createBooleanWrapper(true),
        };
        const { factGraph } = setupFactGraph(datum);
        expect(factGraph.getValue(`/knockoutStudentOrDisabled`, null)).toBe(`true`);
      });

      it(`and spouse is a student, triggers the special earned income knockout`, () => {
        const datum = {
          ...baseData,
          [Path.concretePath(`/filers/*/isDisabled`, primaryFilerId)]: createBooleanWrapper(false),
          [Path.concretePath(`/filers/*/isStudent`, spouseId)]: createBooleanWrapper(true),
          [Path.concretePath(`/secondaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
            createBooleanWrapper(true),
        };
        const { factGraph } = setupFactGraph(datum);
        expect(factGraph.getValue(`/knockoutStudentOrDisabled`, null)).toBe(`true`);
      });
    });

    describe(`if both made < 250`, () => {
      it(`and both were disabled, triggers the special earned income knockout`, () => {
        const datum = {
          ...baseData,
          [Path.concretePath(`/filers/*/isDisabled`, primaryFilerId)]: createBooleanWrapper(true),
          [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(true),
          [Path.concretePath(`/primaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
            createBooleanWrapper(true),
          [Path.concretePath(`/secondaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
            createBooleanWrapper(true),
        };
        const { factGraph } = setupFactGraph(datum);
        expect(factGraph.getValue(`/knockoutStudentOrDisabled`, null)).toBe(`true`);
      });
      it(`and both were students, triggers the special earned income knockout`, () => {
        const datum = {
          ...baseData,
          [Path.concretePath(`/filers/*/isStudent`, primaryFilerId)]: createBooleanWrapper(true),
          [Path.concretePath(`/filers/*/isStudent`, spouseId)]: createBooleanWrapper(true),
          [Path.concretePath(`/primaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
            createBooleanWrapper(true),
          [Path.concretePath(`/secondaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
            createBooleanWrapper(true),
        };
        const { factGraph } = setupFactGraph(datum);
        expect(factGraph.getValue(`/knockoutStudentOrDisabled`, null)).toBe(`true`);
      });
    });
  });

  describe(`with filing status MFS and considered married`, () => {
    const baseData = {
      ...sharedData,
      ...makeFilerData(primaryFilerId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
      [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/filingStatus`]: createEnumWrapper(FilingStatus.MFS, `/filingStatusOptions`),
      ...makeW2Data(50000, w2Id),
      [Path.concretePath(`/formW2s`, null)]: createCollectionWrapper([w2Id]),
    };
    const careProviderUuid = crypto.randomUUID();
    describe(`if tp has and enters spouse earned income and section is filled in`, () => {
      const { factGraph } = setupFactGraph({
        ...baseData,
        [`/flowKnockoutHouseholdEmployee`]: createBooleanWrapper(false),
        '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(true),
        '/writableHasCdccForfeitedCredits': createBooleanWrapper(false),
        '/writableCdccTotalQualifiedDependentCareExpenses': createDollarWrapper(`4000`),
        '/writablePrimaryFilerDependentCarePlanMaximum': createDollarWrapper(`2500`),
        '/writableCdccCarryoverAmountFromPriorTaxYear': createDollarWrapper(`240`),
        '/hasMfjSpouseEarnedIncome': createBooleanWrapper(true),
        '/writableMfsSpouseEarnedIncome': createDollarWrapper(`25000`),
        '/cdccCareProvidersIsDone': createBooleanWrapper(true),
        ...makeCareProviderData(careProviderUuid),
        '/cdccCareProviders': createCollectionWrapper([careProviderUuid]),
      });
      it(`dependent care benefits section is complete`, () => {
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.getValue(`/dependentCareBenefitsIsComplete`, null)).toBe(`true`);
      });
    });
    describe(`if tp reports not having spouse earned income and section is filled in`, () => {
      const { factGraph } = setupFactGraph({
        ...baseData,
        [`/flowKnockoutHouseholdEmployee`]: createBooleanWrapper(false),
        '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(true),
        '/writableHasCdccForfeitedCredits': createBooleanWrapper(false),
        '/writableCdccTotalQualifiedDependentCareExpenses': createDollarWrapper(`4000`),
        '/writablePrimaryFilerDependentCarePlanMaximum': createDollarWrapper(`2500`),
        '/writableCdccCarryoverAmountFromPriorTaxYear': createDollarWrapper(`240`),
        '/hasMfjSpouseEarnedIncome': createBooleanWrapper(false),
        '/cdccCareProvidersIsDone': createBooleanWrapper(true),
        ...makeCareProviderData(careProviderUuid),
        '/cdccCareProviders': createCollectionWrapper([careProviderUuid]),
      });
      it(`dependent care benefits section is complete`, () => {
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.getValue(`/dependentCareBenefitsIsComplete`, null)).toBe(`true`);
      });
    });
    describe(`if tp has spouse earned income and doesn't enter it`, () => {
      const dependentId = `85501abd-9f7c-4746-b4e9-f765565ca923`;

      const qualifyingDependent = {
        ...makeChildData(dependentId, `${CURRENT_TAX_YEAR_AS_NUMBER - 8}-01-01`),
        [`/familyAndHousehold`]: createCollectionWrapper([dependentId]),
        [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(false),
      };
      const { factGraph } = setupFactGraph({
        ...baseData,
        ...qualifyingDependent,
        '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(true),
        '/writableHasCdccForfeitedCredits': createBooleanWrapper(false),
        '/writableCdccTotalQualifiedDependentCareExpenses': createDollarWrapper(`4000`),
        '/writablePrimaryFilerDependentCarePlanMaximum': createDollarWrapper(`2500`),
        '/writableCdccCarryoverAmountFromPriorTaxYear': createDollarWrapper(`240`),
        '/hasMfjSpouseEarnedIncome': createBooleanWrapper(true),
      });
      it(`dependent care benefits section is incomplete`, () => {
        expect(factGraph.get(Path.concretePath(`/taxableDependentCare`, null)).complete.toString()).toBe(`false`);
        expect(factGraph.getValue(`/dependentCareBenefitsIsComplete`, null)).toBe(`false`);
      });
    });
  });
});

describe(`For /cdccSectionIsComplete, a filer`, () => {
  const primaryFilerId = uuid;
  const tpW2Id = `79cb9b1b-7898-429a-8681-2dd64c06a962`;
  const dependentId = `85501abd-9f7c-4746-b4e9-f765565ca923`;

  const incomeRequirementsToQualifySingleData = {
    [`/interestReports`]: createCollectionWrapper([]),
    '/hasForeignAccounts': createBooleanWrapper(false),
    '/isForeignTrustsGrantor': createBooleanWrapper(false),
    '/hasForeignTrustsTransactions': createBooleanWrapper(false),
    [`/wasK12Educators`]: createEnumWrapper(`neither`, `/k12EducatorOptions`),
  };

  const incomeRequirementsToQualifyMFJData = {
    ...incomeRequirementsToQualifySingleData,
    ...livedTogetherFacts,
    [`/socialSecurityReports`]: createCollectionWrapper([]),
    [Path.concretePath(`/filers/*/canBeClaimed`, primaryFilerId)]: createBooleanWrapper(false),
    [Path.concretePath(`/filers/*/canBeClaimed`, spouseId)]: createBooleanWrapper(false),
  };

  const cdccQualifyingDependent = {
    ...makeChildData(dependentId, `${CURRENT_TAX_YEAR_AS_NUMBER - 8}-01-01`),
    [`/familyAndHousehold`]: createCollectionWrapper([dependentId]),
    [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(false),
    [Path.concretePath(`/familyAndHousehold/*/unableToCareForSelf`, dependentId)]: createBooleanWrapper(false),
  };
  const singleFilerWithIncome = {
    ...sharedData,
    ...makeFilerData(primaryFilerId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
    [`/maritalStatus`]: createEnumWrapper(`neverMarried`, `/maritalStatusOptions`),
    [`/filingStatus`]: createEnumWrapper(FilingStatus.SINGLE, `/filingStatusOptions`),
    ...makeW2Data(50000, tpW2Id),
    '/formW2s': {
      $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
      item: { items: [tpW2Id] },
    },
    [`/formW2s/#${tpW2Id}/filer`]: {
      item: {
        id: primaryFilerId,
      },
      $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
    },
  };
  describe(`who is not eligible for CDCC`, () => {
    const { factGraph } = setupFactGraph({
      ...singleFilerWithIncome,
    });
    it(`cdcc credit section is complete`, () => {
      // No qualifying dependents makes /maybeEligibleForCdcc false
      expect(factGraph.get(Path.concretePath(`/maybeEligibleForCdcc`, null)).get.toString()).toBe(`false`);
      expect(factGraph.get(Path.concretePath(`/cdccSectionIsComplete`, null)).get.toString()).toBe(`true`);
      expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).complete.toString()).toBe(`true`);
      expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).get.toString()).toBe(`false`);
    });
    it(`has a disqualifying reason of no qualifying people`, () => {
      const filteredItems = filterConditionalListItems(factGraph, CdccDisqualifyingItems, null);
      expect(filteredItems.length).toBe(1);
      expect(filteredItems[0].itemKey).toBe(`subListCdcc-noQualifyingPersonsNotMFJ`);
    });
    const { factGraph: factGraph2 } = setupFactGraph({
      ...baseFilerData,
      ...makeFilerData(primaryFilerId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
      [`/maritalStatus`]: createEnumWrapper(`neverMarried`, `/maritalStatusOptions`),
      [`/filingStatus`]: createEnumWrapper(FilingStatus.SINGLE, `/filingStatusOptions`),
      [`/filers/#${primaryFilerId}/isDisabled`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
      ...cdccQualifyingDependent,
    });

    it(`who has no earned income but is a student or disabled`, () => {
      expect(factGraph2.get(Path.concretePath(`/maybeEligibleForCdcc`, null)).get.toString()).toBe(`false`);
      expect(factGraph2.get(Path.concretePath(`/cdccHasQualifyingExpenses`, null)).get.toString()).toBe(`false`);
      expect(factGraph2.get(Path.concretePath(`/cdccQualified`, null)).get.toString()).toBe(`false`);
    });
    it(`has a disqualifying reason of no earned income`, () => {
      const filteredItems = filterConditionalListItems(factGraph2, CdccDisqualifyingItems, null);
      expect(filteredItems.length).toBe(1);
      expect(filteredItems[0].itemKey).toBe(`subListCdcc-noEarnedIncome`);
    });
  });
  describe(`is eligible except that their credit would be zero or less due to exclusion benefits`, () => {
    const { factGraph } = setupFactGraph({
      ...singleFilerWithIncome,
      ...cdccQualifyingDependent,
      '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(false),
      [`/formW2s/#${tpW2Id}/writableDependentCareBenefits`]: createDollarWrapper(`4000`),
      [`/formW2s/#${tpW2Id}/filer`]: createCollectionItemWrapper(primaryFilerId),
      '/writableHasCdccForfeitedCredits': createBooleanWrapper(false),
      '/writableCdccTotalQualifiedDependentCareExpenses': createDollarWrapper(`15000`),
      '/writablePrimaryFilerDependentCarePlanMaximum': createDollarWrapper(`5000`),
    });
    it(`cdcc credit section is complete`, () => {
      expect(factGraph.get(Path.concretePath(`/maybeEligibleForCdccBase`, null)).get.toString()).toBe(`true`);
      expect(factGraph.get(Path.concretePath(`/maybeEligibleForCdcc`, null)).get.toString()).toBe(`false`);
      expect(
        factGraph.get(Path.concretePath(`/cdccPotentialCdccIsZeroOrLessDueToExclusionBenefits`, null)).get.toString()
      ).toBe(`true`);
      expect(factGraph.get(Path.concretePath(`/cdccSectionIsComplete`, null)).get.toString()).toBe(`true`);
      expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).complete.toString()).toBe(`true`);
      expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).get.toString()).toBe(`false`);
    });
    it(`has a disqualifying reason of no qualifying expenses`, () => {
      const filteredItems = filterConditionalListItems(factGraph, CdccDisqualifyingItems, null);
      expect(filteredItems.length).toBe(1);
      expect(filteredItems[0].itemKey).toBe(`subListCdcc-metTheExclusionCap`);
    });
  });
  describe(`who is eligible except does not have any expenses`, () => {
    const { factGraph } = setupFactGraph({
      ...singleFilerWithIncome,
      ...cdccQualifyingDependent,
      '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(false),
      '/writableCdccHasQualifyingExpenses': createBooleanWrapper(false),
    });
    it(`cdcc credit section is complete`, () => {
      expect(factGraph.get(Path.concretePath(`/maybeEligibleForCdcc`, null)).get.toString()).toBe(`true`);
      expect(factGraph.get(Path.concretePath(`/cdccSectionIsComplete`, null)).get.toString()).toBe(`true`);
      expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).complete.toString()).toBe(`true`);
      expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).get.toString()).toBe(`false`);
    });
    it(`has a disqualifying reason of no qualifying expenses`, () => {
      const filteredItems = filterConditionalListItems(factGraph, CdccDisqualifyingItems, null);
      expect(filteredItems.length).toBe(1);
      expect(filteredItems[0].itemKey).toBe(`subListCdcc-noQualifyingExpenses`);
    });
  });
  describe(`who is not qualified for CDCC`, () => {
    // Tp is not qualified b/c they didn't pay expenses to a qualified provider
    const { factGraph } = setupFactGraph({
      ...singleFilerWithIncome,
      ...cdccQualifyingDependent,
      ...incomeRequirementsToQualifySingleData,
      '/cdccHasCreditForPriorYearExpenses': createBooleanWrapper(false),
      '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(false),
      '/writableCdccHasQualifyingExpenses': createBooleanWrapper(true),
      [`/familyAndHousehold/#${dependentId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(true),
      [`/familyAndHousehold/#${dependentId}/cdccHadExpensesPaidToQualifyingProvider`]: createBooleanWrapper(false),
    });

    it(`cdcc credit section is complete`, () => {
      expect(factGraph.get(Path.concretePath(`/maybeEligibleForCdcc`, null)).get.toString()).toBe(`true`);
      expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).get.toString()).toBe(`false`);
      expect(factGraph.get(Path.concretePath(`/cdccSectionIsComplete`, null)).get.toString()).toBe(`true`);
      expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).complete.toString()).toBe(`true`);
    });
    it(`has a disqualifying reason of no qualifying expenses`, () => {
      const filteredItems = filterConditionalListItems(factGraph, CdccDisqualifyingItems, null);
      expect(filteredItems.length).toBe(1);
      expect(filteredItems[0].itemKey).toBe(`subListCdcc-noQualifyingExpenses`);
    });
  });
  describe(`who is qualified for CDCC`, () => {
    const qualifyingExpenseData = {
      '/cdccHasCreditForPriorYearExpenses': createBooleanWrapper(false),
      '/hasCdccCarryoverAmountFromPriorTaxYear': createBooleanWrapper(false),
      '/writableCdccHasQualifyingExpenses': createBooleanWrapper(true),
      [`/familyAndHousehold/#${dependentId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(true),
      [`/familyAndHousehold/#${dependentId}/cdccHadExpensesPaidToQualifyingProvider`]: createBooleanWrapper(true),
      [`/familyAndHousehold/#${dependentId}/writableCdccQualifyingExpenseAmount`]: createDollarWrapper(`1300`),
    };

    const careProviderUuid = crypto.randomUUID();

    describe(`who does not have the care providers section completed`, () => {
      it(`cdcc credit section is incomplete`, () => {
        const { factGraph } = setupFactGraph({
          ...singleFilerWithIncome,
          ...cdccQualifyingDependent,
          ...incomeRequirementsToQualifySingleData,
          ...qualifyingExpenseData,
          ...makeCareProviderData(careProviderUuid),
          '/cdccCareProviders': createCollectionWrapper([careProviderUuid]),
          [Path.concretePath(`/cdccCareProvidersIsDone`, null)]: createBooleanWrapper(false),
        });

        expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).get.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/cdccSectionIsComplete`, null)).get.toString()).toBe(`false`);
      });
    });
    describe(`who has the care providers section completed`, () => {
      it(`cdcc credit section is complete`, () => {
        const { factGraph } = setupFactGraph({
          ...singleFilerWithIncome,
          ...cdccQualifyingDependent,
          ...incomeRequirementsToQualifySingleData,
          ...qualifyingExpenseData,
          ...makeCareProviderData(careProviderUuid),
          [Path.concretePath(`/cdccCareProvidersIsDone`, null)]: createBooleanWrapper(true),
          '/cdccCareProviders': createCollectionWrapper([careProviderUuid]),
        });

        expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).get.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/cdccSectionIsComplete`, null)).get.toString()).toBe(`true`);
      });
    });
    describe(`who has to go through the special earned income knockout flow`, () => {
      it(`cdcc credit section is complete`, () => {
        const spouseW2Id = `655db974-7c17-4696-9984-ff709ea49794`;

        const { factGraph } = setupFactGraph({
          ...sharedData,
          ...makeFilerData(primaryFilerId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
          ...makeFilerData(spouseId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
          ...makeW2Data(50000, tpW2Id),
          ...makeW2Data(1000, spouseW2Id),
          [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
          [`/filingStatus`]: createEnumWrapper(FilingStatus.MFJ, `/filingStatusOptions`),
          ...livedTogetherFacts,
          [`/formW2s/#${tpW2Id}/filer`]: createCollectionItemWrapper(primaryFilerId),
          [`/formW2s/#${spouseW2Id}/filer`]: createCollectionItemWrapper(spouseId),
          '/formW2s': createCollectionWrapper([tpW2Id, spouseW2Id]),
          ...cdccQualifyingDependent,
          ...incomeRequirementsToQualifyMFJData,
          ...qualifyingExpenseData,
          [Path.concretePath(`/cdccCareProvidersIsDone`, null)]: createBooleanWrapper(true),
          ...makeCareProviderData(careProviderUuid),
          '/cdccCareProviders': createCollectionWrapper([careProviderUuid]),
          // isStudent opens gate to earned income flow
          [Path.concretePath(`/filers/*/isStudent`, spouseId)]: createBooleanWrapper(true),
          // completes special earned income flow
          [Path.concretePath(`/secondaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`, null)]:
            createBooleanWrapper(false),
        });

        expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).get.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/cdccSectionIsComplete`, null)).get.toString()).toBe(`true`);
      });
    });
    describe(`who has care providers section complete and a non-dep qp`, () => {
      const spouseW2Id = `ec42c7d8-8bfa-4b62-8f7d-346179c797f9`;

      const initialMfjBothEarnedIncomeData = {
        ...sharedData,
        ...makeFilerData(primaryFilerId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
        ...makeFilerData(spouseId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
        ...makeW2Data(50000, tpW2Id),
        ...makeW2Data(1000, spouseW2Id),
        ...cdccQualifyingDependent,
        ...incomeRequirementsToQualifyMFJData,
        ...qualifyingExpenseData,
        [Path.concretePath(`/cdccCareProvidersIsDone`, null)]: createBooleanWrapper(true),
        ...makeCareProviderData(careProviderUuid),
        '/cdccCareProviders': createCollectionWrapper([careProviderUuid]),
        '/formW2s': createCollectionWrapper([tpW2Id, spouseW2Id]),
        [`/formW2s/#${tpW2Id}/filer`]: createCollectionItemWrapper(primaryFilerId),
        [`/formW2s/#${spouseW2Id}/filer`]: createCollectionItemWrapper(spouseId),
      };

      const mfjWithNondepQpData = {
        ...initialMfjBothEarnedIncomeData,
        [Path.concretePath(`/familyAndHousehold/*/unableToCareForSelf`, dependentId)]: createBooleanWrapper(true),
        [Path.concretePath(`/familyAndHousehold/*/dateOfBirth`, dependentId)]: createDayWrapper(`2009-05-15`),
        [Path.concretePath(`/familyAndHousehold/*/married`, dependentId)]: createBooleanWrapper(true),
        [Path.concretePath(`/familyAndHousehold/*/writableJointReturn`, dependentId)]: createBooleanWrapper(true),
        [Path.concretePath(`/familyAndHousehold/*/writableRequiredToFile`, dependentId)]: createBooleanWrapper(true),
      };

      it(`cdcc credit section is incomplete without tin collected`, () => {
        const { factGraph } = setupFactGraph({
          ...mfjWithNondepQpData,
          ...incomeRequirementsToQualifyMFJData,
        });

        expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).get.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/cdccNonDepQpTinsNoneToCollect`, null)).get.toString()).toBe(`false`);
        expect(factGraph.get(Path.concretePath(`/cdccSectionIsComplete`, null)).get.toString()).toBe(`false`);
      });

      it(`cdcc credit section is complete when tins and pins have been collected`, () => {
        // the tin is already set in the data, but without the other facts below, the tin info is not complete
        const { factGraph } = setupFactGraph({
          ...mfjWithNondepQpData,
          ...incomeRequirementsToQualifyMFJData,
          [Path.concretePath(`/cdccCareProvidersIsDone`, null)]: createBooleanWrapper(true),
          ...makeCareProviderData(careProviderUuid),
          [`/familyAndHousehold/#${dependentId}/tin`]: createTinWrapper({ area: `555`, group: `00`, serial: `5555` }),
          [`/familyAndHousehold/#${dependentId}/hasIpPin`]: createBooleanWrapper(false),
        });

        expect(
          factGraph
            .get(Path.concretePath(`/familyAndHousehold/*/cdccNonDependentQualifyingPerson`, dependentId))
            .get.toString()
        ).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).complete.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/cdccQualified`, null)).get.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/cdccNonDepQpTinsNoneToCollect`, null)).get.toString()).toBe(`true`);
        expect(factGraph.get(Path.concretePath(`/cdccSectionIsComplete`, null)).get.toString()).toBe(`true`);
      });
    });
  });
});

describe(`Special Earned Income Rule, taxpayer with a cdcc qualifying person`, () => {
  const primaryFilerId = uuid;
  const spouseId = `859c03d1-af4a-447f-96aa-d19397048a48`;
  const tpW2Id = `79cb9b1b-7898-429a-8681-2dd64c06a962`;
  const spouseW2Id = `3d56a54d-09cc-4f9f-abd5-6dd48508bf95`;
  const dependentId = `85501abd-9f7c-4746-b4e9-f765565ca923`;

  const makeQualifyingDependent = (dependentId: string) => {
    return {
      ...makeChildData(dependentId, `${CURRENT_TAX_YEAR_AS_NUMBER - 8}-01-01`),
      [`/familyAndHousehold`]: createCollectionWrapper([dependentId]),
      [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(false),
    };
  };

  const makeInitialMFJWithQpData = (spouseIncome: number) => {
    return {
      ...sharedData,
      ...makeFilerData(primaryFilerId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
      ...makeFilerData(spouseId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
      ...livedTogetherFacts,
      [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/filingStatus`]: createEnumWrapper(FilingStatus.MFJ, `/filingStatusOptions`),
      ...makeW2Data(50000, tpW2Id),
      ...makeW2Data(spouseIncome, spouseW2Id),
      '/formW2s': createCollectionWrapper([tpW2Id, spouseW2Id]),
      [`/formW2s/#${tpW2Id}/filer`]: createCollectionItemWrapper(primaryFilerId),
      [`/formW2s/#${spouseW2Id}/filer`]: createCollectionItemWrapper(spouseId),
      ...makeQualifyingDependent(dependentId),
    };
  };

  const makeDependentCareBenefitsFacts = (w2Id = spouseW2Id) => {
    return {
      [`/formW2s/#${w2Id}/writableDependentCareBenefits`]: createDollarWrapper(`500.00`),
      [`/writableCdccTotalQualifiedDependentCareExpenses`]: createDollarWrapper(`25000`),
      [`/writableSecondaryFilerDependentCarePlanMaximum`]: createDollarWrapper(`5000`),
      [`/writablePrimaryFilerDependentCarePlanMaximum`]: createDollarWrapper(`5000`),
    };
  };

  const dependentIsQpFacts = {
    [`/familyAndHousehold/#${dependentId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(true),
    [`/familyAndHousehold/#${dependentId}/cdccHadExpensesPaidToQualifyingProvider`]: createBooleanWrapper(true),
    [`/familyAndHousehold/#${dependentId}/writableCdccQualifyingExpenseAmount`]: createDollarWrapper(`21000`),
  };

  describe(`who may be eligible for CDCC special income rule`, () => {
    describe(`who is MFJ with student spouse and benefits`, () => {
      it(`may see the special income knockout flow`, () => {
        //   tests /specialEarnedIncomeFlowRequirementsSansIncomeAndFilingStatus
        const data = {
          ...makeInitialMFJWithQpData(1000),
          ...makeDependentCareBenefitsFacts(),
          ...dependentIsQpFacts,
          [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(true),
          [`/filers/#${spouseId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(false),
        };
        const { factGraph } = setupFactGraph(data);
        const fact = factGraph.get(
          Path.concretePath(`/specialEarnedIncomeFlowRequirementsSansIncomeAndFilingStatus`, null)
        );
        expect(fact.get).toBe(true);
      });
      describe(`will see knockout in the benefits subsection (and not the credits subsection)`, () => {
        it(`when had benefits and lower earning filer made less than 5k`, () => {
          const data = {
            ...makeInitialMFJWithQpData(1000),
            ...makeDependentCareBenefitsFacts(),
            [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(true),
            [`/filers/#${spouseId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(false),
          };
          const { factGraph } = setupFactGraph(data);
          const showInBenefitsFact = factGraph.get(
            Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)
          );
          const showInCreditsFact = factGraph.get(
            Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInCredits`, null)
          );
          expect(showInBenefitsFact.get).toBe(true);
          expect(showInCreditsFact.get).toBe(false);
        });
      });
      describe(`will see the knockout in neither benefits nor the credits section`, () => {
        it(`when lower income >= 6k`, () => {
          const data = {
            ...makeInitialMFJWithQpData(6000),
            ...makeDependentCareBenefitsFacts(),
            [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(true),
            [`/filers/#${spouseId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(false),
          };
          const { factGraph } = setupFactGraph(data);
          const showInBenefitsFact = factGraph.get(
            Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)
          );
          const showInCreditsFact = factGraph.get(
            Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInCredits`, null)
          );
          expect(showInBenefitsFact.get).toBe(false);
          expect(showInCreditsFact.get).toBe(false);
        });
      });
      describe(`will see knockout in the credits subsection and not the benefits subsection`, () => {
        it(`when lower earned income < 3000, no benefits and one qualifying person`, () => {
          const data = {
            ...makeInitialMFJWithQpData(2750),
            ...dependentIsQpFacts,
            [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(true),
            [`/filers/#${spouseId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(false),
          };
          const { factGraph } = setupFactGraph(data);

          const showInBenefitsFact = factGraph.get(
            Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)
          );
          const showInCreditsFact = factGraph.get(
            Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInCredits`, null)
          );
          expect(showInBenefitsFact.get).toBe(false);
          expect(showInCreditsFact.get).toBe(true);
        });

        it(`when they had benefits, two qualifying people, and had between 5-6k income`, () => {
          const data = {
            ...makeInitialMFJWithQpData(5500),
            // below income ensures filer with two qps (2ndaryfiler + dependent) and lower earned income between 5-6k
            // sees it in credits
            [`/filers/#${spouseId}/isDisabled`]: createBooleanWrapper(true),
            [`/filers/#${spouseId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(false),
            [`/familyAndHousehold/#${dependentId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(true),
            [`/familyAndHousehold/#${dependentId}/cdccHadExpensesPaidToQualifyingProvider`]: createBooleanWrapper(true),
            [`/familyAndHousehold/#${dependentId}/writableCdccQualifyingExpenseAmount`]: createDollarWrapper(`21000`),
            // includes dependent care benefits
            [`/formW2s/#${spouseW2Id}/writableDependentCareBenefits`]: createDollarWrapper(`500.00`),
            [`/writableCdccTotalQualifiedDependentCareExpenses`]: createDollarWrapper(`25000`),
            [`/writableSecondaryFilerDependentCarePlanMaximum`]: createDollarWrapper(`5000`),
          };
          const { factGraph } = setupFactGraph(data);

          const showInBenefitsFact = factGraph.get(
            Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)
          );
          const showInCreditsFact = factGraph.get(
            Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInCredits`, null)
          );
          expect(showInBenefitsFact.get).toBe(false);
          expect(showInCreditsFact.get).toBe(true);
        });
      });
      describe(`will not see the knockout in the credits subsection`, () => {
        it(`when MFJ lower spouse income >= 6k and two qualified people`, () => {
          const data = {
            ...makeInitialMFJWithQpData(6000),
            ...dependentIsQpFacts,
            [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(true),
            // creates second qp
            [`/filers/#${spouseId}/isDisabled`]: createBooleanWrapper(true),
            [`/filers/#${spouseId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(false),
          };
          const { factGraph } = setupFactGraph(data);

          const showInBenefitsFact = factGraph.get(
            Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)
          );
          const showInCreditsFact = factGraph.get(
            Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInCredits`, null)
          );
          expect(showInBenefitsFact.get).toBe(false);
          expect(showInCreditsFact.get).toBe(false);
        });
        it(`when MFJ lower spouse income > 3k and one qualified people`, () => {
          const data = {
            ...makeInitialMFJWithQpData(3000),
            ...dependentIsQpFacts,
            [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(true),
            [`/filers/#${spouseId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(false),
          };
          const { factGraph } = setupFactGraph(data);

          const showInBenefitsFact = factGraph.get(
            Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)
          );
          const showInCreditsFact = factGraph.get(
            Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInCredits`, null)
          );
          expect(showInBenefitsFact.get).toBe(false);
          expect(showInCreditsFact.get).toBe(false);
        });
      });
    });
  });
  describe(`who is MFS`, () => {
    const makeInitialMFSWithQpData = (income: number) => {
      return {
        ...sharedData,
        ...makeFilerData(primaryFilerId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
        [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
        [`/filingStatus`]: createEnumWrapper(FilingStatus.MFS, `/filingStatusOptions`),
        ...makeW2Data(income, tpW2Id),
        '/formW2s': createCollectionWrapper([tpW2Id]),
        [`/formW2s/#${tpW2Id}/filer`]: createCollectionItemWrapper(primaryFilerId),
        ...makeQualifyingDependent(dependentId),
        [`/familyAndHousehold/#${dependentId}/tpPaidMostOfHomeUpkeep`]: createBooleanWrapper(true),
      };
    };

    describe(`will not qualify for credit lived apart exception not met`, () => {
      it(`if MFS but didn't live apart for last six months`, () => {
        const data = {
          ...makeInitialMFSWithQpData(1000),
          ...makeDependentCareBenefitsFacts(tpW2Id),
          ...livedApartFacts,
          // overwrite lived apart exception
          [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(false),
        };
        const { factGraph } = setupFactGraph(data);
        expect(factGraph.get(Path.concretePath(`/mfsButEligibleForCdcc`, null)).get).toBe(false);
        expect(factGraph.get(Path.concretePath(`/mfsNotEligibleForCdcc`, null)).get).toBe(true);
        expect(factGraph.get(Path.concretePath(`/maybeEligibleForCdcc`, null)).get).toBe(false);
      });

      it.skip(`if MFS but didn't upkeep home`, () => {
        const data = {
          ...makeInitialMFSWithQpData(1000),
          ...makeDependentCareBenefitsFacts(tpW2Id),
          ...livedApartFacts,
          // overwrite home upkeep exception
          [`/familyAndHousehold/#${dependentId}/tpPaidMostOfHomeUpkeep`]: createBooleanWrapper(false),
        };
        const { factGraph } = setupFactGraph(data);
        expect(factGraph.get(Path.concretePath(`/mfsButEligibleForCdcc`, null)).get).toBe(false);
        expect(factGraph.get(Path.concretePath(`/mfsNotEligibleForCdcc`, null)).get).toBe(true);
        // TODO should be false but is true
        expect(factGraph.get(Path.concretePath(`/maybeEligibleForCdcc`, null)).get).toBe(false);
      });

      it(`if MFS but your home was not qps main home for more than half the year`, () => {
        const data = {
          ...makeInitialMFSWithQpData(1000),
          ...makeDependentCareBenefitsFacts(tpW2Id),
          ...livedApartFacts,
          [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(true),
          [`/filers/#${spouseId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(false),
          // overwrite qualifying person's main home for more than half the year
          [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
            `lessThanSixMonths`,
            `/residencyDurationOptions`
          ),
        };
        const { factGraph } = setupFactGraph(data);
        expect(factGraph.get(Path.concretePath(`/mfsButEligibleForCdcc`, null)).get).toBe(false);
        expect(factGraph.get(Path.concretePath(`/mfsNotEligibleForCdcc`, null)).get).toBe(true);
        expect(factGraph.get(Path.concretePath(`/maybeEligibleForCdcc`, null)).get).toBe(false);
      });
    });

    describe(`will see knockout in the benefits subsection and not the credits subsection`, () => {
      it(`if MFS considered married and earned less than 2.5k`, () => {
        const data = {
          ...makeInitialMFSWithQpData(1000),
          ...makeDependentCareBenefitsFacts(tpW2Id),
          [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(true),
          [`/filers/#${spouseId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(false),
          ...livedTogetherFacts,
        };
        const { factGraph } = setupFactGraph(data);

        const showInBenefitsFact = factGraph.get(
          Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)
        );
        const showInCreditsFact = factGraph.get(
          Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInCredits`, null)
        );

        expect(showInBenefitsFact.get).toBe(true);
        expect(showInCreditsFact.get).toBe(false);
      });
    });
    describe(`will not see the knockout in the benefits section or the credits section`, () => {
      it(`when MFS considered unmarried`, () => {
        const data = {
          ...makeInitialMFSWithQpData(1000),
          ...makeDependentCareBenefitsFacts(tpW2Id),
          [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(true),
          [`/filers/#${spouseId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(false),
          ...livedApartFacts,
        };
        const { factGraph } = setupFactGraph(data);

        expect(factGraph.get(Path.concretePath(`/mfsButEligibleForCdcc`, null)).get).toBe(true);

        const showInBenefitsFact = factGraph.get(
          Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)
        );
        const showInCreditsFact = factGraph.get(
          Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInCredits`, null)
        );

        expect(showInBenefitsFact.get).toBe(false);
        expect(showInCreditsFact.get).toBe(false);
      });
    });

    describe(`will not see the knockout in the credits subsection`, () => {
      it(`if MFS considered married, had no benefits, and earned less than 2.5k`, () => {
        const data = {
          ...makeInitialMFSWithQpData(1000),
          [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(true),
          [`/filers/#${spouseId}/cdccHasDependentCareExpenses`]: createBooleanWrapper(false),
          ...livedTogetherFacts,
        };
        const { factGraph } = setupFactGraph(data);

        const showInBenefitsFact = factGraph.get(
          Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInBenefits`, null)
        );
        const showInCreditsFact = factGraph.get(
          Path.concretePath(`/showSpecialEarnedIncomeKnockoutFlowInCredits`, null)
        );

        expect(showInBenefitsFact.get).toBe(false);
        expect(showInCreditsFact.get).toBe(false);
      });
    });
  });
  describe(`who see the knockout flow`, () => {
    const showKoFlowData = {
      ...makeInitialMFJWithQpData(1000),
      [`/formW2s/#${spouseW2Id}/writableDependentCareBenefits`]: createDollarWrapper(`500.00`),
      [`/writableSecondaryFilerDependentCarePlanMaximum`]: createDollarWrapper(`5000`),
      [`/writableCdccTotalQualifiedDependentCareExpenses`]: createDollarWrapper(`25000`),
    };
    it.each([
      {
        msg: `shows ko-determining screen for tp1 only`,
        tp1IsDisabled: true,
        tp1IsStudent: false,
        tp2IsDisabled: false,
        tp2IsStudent: false,
        showTp1Screen: true,
        showTp2Screen: false,
      },
      {
        msg: `shows ko-determining screen for tp2 only`,
        tp1IsDisabled: false,
        tp1IsStudent: false,
        tp2IsDisabled: false,
        tp2IsStudent: true,
        showTp1Screen: false,
        showTp2Screen: true,
      },
      {
        msg: `shows ko-determining screen for tp1 and tp2`,
        tp1IsDisabled: true,
        tp1IsStudent: false,
        tp2IsDisabled: false,
        tp2IsStudent: true,
        tp1Answer: false,
        showTp1Screen: true,
        showTp2Screen: true,
      },
      {
        msg: `shows neither screen`,
        tp1IsDisabled: false,
        tp1IsStudent: false,
        tp2IsDisabled: false,
        tp2IsStudent: false,
        showTp1Screen: false,
        showTp2Screen: false,
      },
    ])(`%s`, (scenario) => {
      let data;
      const preData = {
        ...showKoFlowData,
        [`/filers/#${primaryFilerId}/isDisabled`]: createBooleanWrapper(scenario.tp1IsDisabled),
        [`/filers/#${primaryFilerId}/isStudent`]: createBooleanWrapper(scenario.tp1IsStudent),

        [`/filers/#${spouseId}/isDisabled`]: createBooleanWrapper(scenario.tp2IsDisabled),
        [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(scenario.tp2IsStudent),
      };
      if (scenario?.tp1Answer) {
        data = {
          ...preData,
          [`/primaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`]: createBooleanWrapper(true),
        };
      } else if (scenario?.tp1Answer === false) {
        data = {
          ...preData,
          [`/primaryFilerMadeLessThanDisabledOrStudentMonthlyIncome`]: createBooleanWrapper(false),
        };
      } else {
        data = preData;
      }
      const { factGraph } = setupFactGraph(data);
      const showTp1ScreenFact = factGraph.get(Path.concretePath(`/primaryFiler/isStudentOrDisabled`, null));
      const showTp2ScreenFact = factGraph.get(Path.concretePath(`/showEarnedIncomeRuleTp2`, null));
      expect(showTp1ScreenFact.get).toBe(scenario.showTp1Screen);
      expect(showTp2ScreenFact.get).toBe(scenario.showTp2Screen);
    });
  });
});

describe(`Allocation of taxable dependent care benefits, taxpayer who is MFJ with CDCC qualifying person`, () => {
  const primaryFilerId = uuid;
  const spouseId = `859c03d1-af4a-447f-96aa-d19397048a48`;
  const tpW2Id = `79cb9b1b-7898-429a-8681-2dd64c06a962`;
  const spouseW2Id = `3d56a54d-09cc-4f9f-abd5-6dd48508bf95`;
  const dependentId = `85501abd-9f7c-4746-b4e9-f765565ca923`;

  const qualifyingDependent = {
    ...makeChildData(dependentId, `${CURRENT_TAX_YEAR_AS_NUMBER - 8}-01-01`),
    [`/familyAndHousehold`]: createCollectionWrapper([dependentId]),
    [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(false),
  };
  const initialData = {
    ...sharedData,
    ...makeFilerData(primaryFilerId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
    ...makeFilerData(spouseId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
    [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
    [`/filingStatus`]: createEnumWrapper(FilingStatus.MFJ, `/filingStatusOptions`),
    ...makeW2Data(50000, tpW2Id),
    ...makeW2Data(1000, spouseW2Id),
    ...qualifyingDependent,
    '/formW2s': createCollectionWrapper([tpW2Id, spouseW2Id]),
    [`/formW2s/#${tpW2Id}/filer`]: {
      item: {
        id: primaryFilerId,
      },
      $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
    },
    [`/formW2s/#${spouseW2Id}/filer`]: {
      item: {
        id: spouseId,
      },
      $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
    },
  };
  describe(`whose household member was unable to care for themselves who would otherwise be a dependent`, () => {
    const dataWithEligibleDependent = {
      ...initialData,
      [Path.concretePath(`/familyAndHousehold/*/unableToCareForSelf`, dependentId)]: createBooleanWrapper(true),
    };
    const qrGrossIncomeLimit = 4700;
    it(`is eligible as a dependent`, () => {
      const { factGraph } = setupFactGraph(dataWithEligibleDependent);
      const eligibleDependentFact = factGraph.get(
        Path.concretePath(`/familyAndHousehold/*/eligibleDependent`, dependentId)
      );

      const cdccNondepQpFact = factGraph.get(
        Path.concretePath(`/familyAndHousehold/*/cdccNonDependentQualifyingPerson`, dependentId)
      );
      const cdccQpFact = factGraph.get(Path.concretePath(`/familyAndHousehold/*/cdccQualifyingPerson`, dependentId));

      expect(eligibleDependentFact.get).toBe(true);
      expect(cdccNondepQpFact.get).toBe(false);
      expect(cdccQpFact.get).toBe(true);
    });

    it(`except they had gross income of $${qrGrossIncomeLimit} or more is a qualifying person`, () => {
      const data = {
        ...dataWithEligibleDependent,
        // below sets up a qualifying relative for whom we need to know gross income:
        [Path.concretePath(`/familyAndHousehold/*/dateOfBirth`, dependentId)]: createDayWrapper(`1995-05-15`),
        [Path.concretePath(`/familyAndHousehold/*/permanentTotalDisability`, dependentId)]: createBooleanWrapper(false),
        [Path.concretePath(`/familyAndHousehold/*/writableCouldBeQualifyingChildOfAnother`, dependentId)]:
          createBooleanWrapper(true),
        [Path.concretePath(`/familyAndHousehold/*/ownSupport`, dependentId)]: createBooleanWrapper(false),
        [Path.concretePath(`/familyAndHousehold/*/writableQrSupportTest`, dependentId)]: createBooleanWrapper(true),
        // the condition that prevents them from being a dependent:
        [Path.concretePath(`/familyAndHousehold/*/grossIncomeTest`, dependentId)]: createBooleanWrapper(false),
      };
      const { factGraph } = setupFactGraph(data);

      const eligibleDependentFact = factGraph.get(
        Path.concretePath(`/familyAndHousehold/*/eligibleDependent`, dependentId)
      );
      const cdccNonDepQpFact = factGraph.get(
        Path.concretePath(`/familyAndHousehold/*/cdccNonDependentQualifyingPerson`, dependentId)
      );
      const cdccQpFact = factGraph.get(Path.concretePath(`/familyAndHousehold/*/cdccQualifyingPerson`, dependentId));

      expect(eligibleDependentFact.get).toBe(false);
      expect(cdccNonDepQpFact.get).toBe(true);
      expect(cdccQpFact.get).toBe(true);
    });
    it(`except they filed a joint return is a qualifying person`, () => {
      const data = {
        ...dataWithEligibleDependent,
        [Path.concretePath(`/familyAndHousehold/*/dateOfBirth`, dependentId)]: createDayWrapper(`2009-05-15`),
        [Path.concretePath(`/familyAndHousehold/*/married`, dependentId)]: createBooleanWrapper(true),
        [Path.concretePath(`/familyAndHousehold/*/writableJointReturn`, dependentId)]: createBooleanWrapper(true),
        [Path.concretePath(`/familyAndHousehold/*/writableRequiredToFile`, dependentId)]: createBooleanWrapper(true),
      };

      const { factGraph } = setupFactGraph(data);
      const eligibleDependentFact = factGraph.get(
        Path.concretePath(`/familyAndHousehold/*/eligibleDependent`, dependentId)
      );
      const cdccNonDepQpFact = factGraph.get(
        Path.concretePath(`/familyAndHousehold/*/cdccNonDependentQualifyingPerson`, dependentId)
      );
      const cdccQpFact = factGraph.get(Path.concretePath(`/familyAndHousehold/*/cdccQualifyingPerson`, dependentId));

      expect(eligibleDependentFact.get).toBe(false);
      expect(cdccNonDepQpFact.get).toBe(true);
      expect(cdccQpFact.get).toBe(true);
    });

    it(`TP (or their spouse if filing jointly) could be claimed as a dependent on another taxpayer's 2023 return
    is a qualifying person`, () => {
      const data = {
        ...dataWithEligibleDependent,
        [Path.concretePath(`/familyAndHousehold/*/dateOfBirth`, dependentId)]: createDayWrapper(`2019-05-15`),
        [Path.concretePath(`/filers/*/canBeClaimed`, spouseId)]: createBooleanWrapper(true),
        // isMFJDependent:
        [Path.concretePath(`/MFJRequiredToFile`, null)]: createBooleanWrapper(false),
        [Path.concretePath(`/MFJDependentsFilingForCredits`, null)]: createBooleanWrapper(false),
      };
      const { factGraph } = setupFactGraph(data);

      const eligibleDependentFact = factGraph.get(
        Path.concretePath(`/familyAndHousehold/*/eligibleDependent`, dependentId)
      );
      const cdccNonDepQpFact = factGraph.get(
        Path.concretePath(`/familyAndHousehold/*/cdccNonDependentQualifyingPerson`, dependentId)
      );
      const cdccQpFact = factGraph.get(Path.concretePath(`/familyAndHousehold/*/cdccQualifyingPerson`, dependentId));

      expect(factGraph.get(Path.concretePath(`/treatAsMFJ`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/flowMFJSpouseFilingRequirementSubsection`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/MFJClaimingRefundOnly`, null)).get).toBe(true);
      expect(factGraph.get(Path.concretePath(`/MFJRequiredToFile`, null)).get).toBe(false);
      expect(factGraph.get(Path.concretePath(`/treatFilersAsDependents`, null)).get).toBe(true);
      expect(eligibleDependentFact.get).toBe(false);
      expect(cdccNonDepQpFact.get).toBe(true);
      expect(cdccQpFact.get).toBe(true);
    });
  });
});

describe(`An MFJ filer`, () => {
  const dependentId = crypto.randomUUID();
  const primaryW2Id = crypto.randomUUID();
  const secondaryW2Id = crypto.randomUUID();

  const spouseData = makeFilerData(spouseId, { [`/canBeClaimed`]: createBooleanWrapper(false) });
  const baseData = {
    ...sharedData,
    ...livedTogetherFacts,
    ...spouseData,
    [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
    [`/filingStatus`]: createEnumWrapper(FilingStatus.MFJ, `/filingStatusOptions`),
    ...makeClaimedChild(dependentId, {
      [`/dateOfBirth`]: createDayWrapper(`${CURRENT_TAX_YEAR_AS_NUMBER - 10}-01-01`),
      [`/tpClaims`]: createBooleanWrapper(true),
    }),
    [`/familyAndHousehold`]: createCollectionWrapper([dependentId]),
    [`/writablePrimaryFilerDependentCarePlanMaximum`]: createDollarWrapper(`5000.00`),
    [`/writableSecondaryFilerDependentCarePlanMaximum`]: createDollarWrapper(`5000.00`),
    [`/flowKnockoutHouseholdEmployee`]: createBooleanWrapper(false),
    [`/writableHasCdccForfeitedCredits`]: createBooleanWrapper(false),
    [`/hasCdccCarryoverAmountFromPriorTaxYear`]: createBooleanWrapper(false),
  };

  describe.each([
    // Descriptions for each case provided by SME and lightly edited for convenience
    /*
      Suppose A&B are MFJ taxpayers, and
      A has $50k in earned income and dependent care benefits of $3k through A’s employer
      B has $2k in earned income and dependent care benefits of $2k through B’s employer.
      A&B incurred childcare expenses of $5k but are limited to $2k under the rules so $3k becomes taxable benefits.

      It seems that you could add an additional $2k to taxpayer B’s income, which would bring the earned income
      limitation to $4k.

      I don’t believe you could add the full $3k to B because B didn’t have the opportunity to make $5k (as B had
      earned income of $2k initially plus $2k in dependent care benefits bringing B’s total taxable income to $4k).

      Now, for purposes of calculating the CDCC, you could use $3k in expenses with an earned income limitation of
      $4k so it seems like the taxpayers could get the full credit here.
    */
    {
      primary: {
        earnedIncome: 50_000,
        dependentCareBenefits: 3_000,
        earnedIncomeAdjustment: 1_000,
        shouldGetFullAllocation: false,
      },
      secondary: {
        earnedIncome: 2_000,
        dependentCareBenefits: 2_000,
        earnedIncomeAdjustment: 2_000,
        shouldGetFullAllocation: true,
      },
      qualifiedExpenses: 5_000,
      taxableDependentCareBenefits: 3_000,
    },
    /*
      Suppose A&B are MFJ taxpayers, and
      A has $50k in earned income and dependent care benefits of $5k through A’s employer
      B has $2k in earned income and dependent care benefits of $1k through B’s employer.
      A&B incurred childcare expenses of $6k but are limited to $2k under the rules so $4k becomes taxable benefits.

      It seems that you could add an additional $1k to taxpayer B’s income, which would bring the earned income
      limitation to $3k.

      Again, I don’t believe you could add the full $4k to B because B didn’t have the opportunity to make $4k (as B
      had earned income of $2k initially plus $1k in dependent care benefits bringing B’s total potential taxable
      income to $3k).

      Now, for purposes of calculating the CDCC, you could use $4k in expenses with an earned income limitation of
      $3k so it seems like you would not get full credit for expenses here.
    */
    {
      primary: {
        earnedIncome: 50_000,
        dependentCareBenefits: 5_000,
        earnedIncomeAdjustment: 3_000,
        shouldGetFullAllocation: false,
      },
      secondary: {
        earnedIncome: 2_000,
        dependentCareBenefits: 1_000,
        earnedIncomeAdjustment: 1_000,
        shouldGetFullAllocation: true,
      },
      qualifiedExpenses: 6_000,
      taxableDependentCareBenefits: 4_000,
    },
    /*
      Suppose A&B are MFJ taxpayers,
      A has $50k in earned income and dependent care benefits of $5k through A’s employer.
      B does not work at all (and does not fall under any special rule for earned income).

      Now, A would have $5k in taxable dependent care benefits.
      You cannot treat this as $5k in taxable income to B since B was not the employee and therefore A would be
      the one who would have earned this income.
      I don’t think there’s any eligibility for a CDCC under the rules in this scenario in addition to no
      exclusion for A.
      (This could be a more common scenario).
    */
    {
      primary: {
        earnedIncome: 50_000,
        dependentCareBenefits: 5_000,
        earnedIncomeAdjustment: 5_000,
        shouldGetFullAllocation: true,
      },
      secondary: { earnedIncome: 0, earnedIncomeAdjustment: 0, shouldGetFullAllocation: false },
      qualifiedExpenses: 5_000,
      taxableDependentCareBenefits: 5_000,
    },
    /*
    Suppose A&B are MFJ taxpayers,
    A has $2000 in earned income and $3000 in dependent care benefits through their employer.
    B also has $2000 in earned income and $3000 in dependent care benefits through their employer.
    They have a single qualifying dependent, and $6000 in total qualifying expenses for them.
    Their total taxable benefits would then be $3000.

    We _could_ add all of this as taxable earned income to the primary filer such that they have $5000 in earned income.
    However, their maximum credit would be bottlenecked by the secondary filer's earned income of $2000. So, we would
    require additional logic to capture this situation -- which currently doesn't exist.
    This is an uncommon scenario, but in this case `shouldGetFullAllocation`s for both filers should be false.

    The filers having a non-zero earned-income adjustment is to be expected due to logic that adds the remainder of the
    (expense cap - other filer's max benefit) if all if they cannot get the full allocation
    */
    {
      primary: {
        earnedIncome: 2_000,
        dependentCareBenefits: 3_000,
        earnedIncomeAdjustment: 1_000,
        shouldGetFullAllocation: false,
      },
      secondary: {
        earnedIncome: 2_000,
        dependentCareBenefits: 3_000,
        earnedIncomeAdjustment: 1_000,
        shouldGetFullAllocation: false,
      },
      qualifiedExpenses: 5_000,
      taxableDependentCareBenefits: 4_000,
    },
  ])(
    `with $primary.earnedIncome in earned income and ` +
      `dependent care benefits of $primary.dependentCareBenefits through employer\n\t` +
      `and a secondary filer with $secondary.earnedIncome in earned income and ` +
      `dependent care benefits of $secondary.dependentCareBenefits through employer\n\t` +
      `and $qualifiedExpenses in qualified expenses\n\t` +
      `and $taxableDependentCareBenefits worth of taxable dependent care benefits`,
    ({ primary, secondary, qualifiedExpenses, taxableDependentCareBenefits }) => {
      it(`allocates ${primary.earnedIncomeAdjustment ?? secondary.earnedIncomeAdjustment ?? 0} to the ${
        primary.earnedIncomeAdjustment > 0 ? `primary` : `secondary`
      } filer's earned income`, () => {
        const data = {
          ...baseData,
          ...makeW2Data(primary.earnedIncome, primaryW2Id, primaryFilerId),
          ...makeW2Data(secondary.earnedIncome, secondaryW2Id, spouseId),
          [`/formW2s`]: createCollectionWrapper([primaryW2Id, secondaryW2Id]),
          [Path.concretePath(`/formW2s/*/writableDependentCareBenefits`, primaryW2Id)]: createDollarWrapper(
            (primary.dependentCareBenefits ?? 0).toString()
          ),
          [Path.concretePath(`/formW2s/*/writableDependentCareBenefits`, secondaryW2Id)]: createDollarWrapper(
            (secondary.dependentCareBenefits ?? 0).toString()
          ),
          [Path.concretePath(`/writableCdccTotalQualifiedDependentCareExpenses`, null)]: createDollarWrapper(
            qualifiedExpenses.toString()
          ),
          [Path.concretePath(`/familyAndHousehold/*/writableCdccQualifyingExpenseAmount`, dependentId)]:
            createDollarWrapper(qualifiedExpenses.toString()),
        };
        const { factGraph } = setupFactGraph(data);

        // Split allocation is not supported, so the allocation must always be all or nothing
        expect(
          factGraph.get(`/cdccPrimaryFilerMaximumEarnedIncomeAdjustmentFromTaxableBenefits` as ConcretePath).complete
        ).toBe(true);
        const allocateToPrimaryFact = factGraph.get(
          `/cdccAllTaxableDependentCareBenefitsShouldBeAddedtoPrimaryEarnedIncome` as ConcretePath
        );
        expect(allocateToPrimaryFact.complete).toBe(true);
        expect(allocateToPrimaryFact.get).toBe(primary.shouldGetFullAllocation);

        const allocateToSecondaryFact = factGraph.get(
          `/cdccAllTaxableDependentCareBenefitsShouldBeAddedtoSecondaryEarnedIncome` as ConcretePath
        );
        expect(allocateToSecondaryFact.complete).toBe(true);
        expect(allocateToSecondaryFact.get).toBe(secondary.shouldGetFullAllocation);

        // Taxable Dependent Care Benefits
        const taxableDependentCareBenefitsFact = factGraph.get(Path.concretePath(`/cdccTaxableBenefits`, null));
        expect(taxableDependentCareBenefitsFact.complete).toBe(true);
        expect(taxableDependentCareBenefitsFact.get.toString()).toBe(taxableDependentCareBenefits.toFixed(2));

        // Primary Allocation
        const primaryFilerTaxableBenefitsAllocation = factGraph.get(
          Path.concretePath(`/cdccPrimaryFilerTaxableBenefits`, null)
        );
        expect(primaryFilerTaxableBenefitsAllocation.complete).toBe(true);
        expect(primaryFilerTaxableBenefitsAllocation.get.toString()).toBe(
          (primary.earnedIncomeAdjustment ?? 0).toFixed(2)
        );

        // SEcondary Allocation
        const secondaryFilerTaxableBenefitsAllocation = factGraph.get(
          Path.concretePath(`/cdccSecondaryFilerTaxableBenefits`, null)
        );
        expect(secondaryFilerTaxableBenefitsAllocation.complete).toBe(true);
        expect(secondaryFilerTaxableBenefitsAllocation.get.toString()).toBe(
          (secondary.earnedIncomeAdjustment ?? 0).toFixed(2)
        );
      });
    }
  );
});

describe(`A /cdccCareProvider collection item`, () => {
  const dependentId = crypto.randomUUID();
  const providerId = crypto.randomUUID();

  const baseData = {
    ...sharedData,
    ...makeFilerData(primaryFilerId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
    [`/maritalStatus`]: createEnumWrapper(`single`, `/maritalStatusOptions`),
    [`/filingStatus`]: createEnumWrapper(FilingStatus.SINGLE, `/filingStatusOptions`),
    ...makeClaimedChild(dependentId, {
      [`/unableToCareForSelf`]: createBooleanWrapper(true),
      [`/residencyDuration`]: createEnumWrapper(`allYear`, `residencyDurationOptions`),
      [`/tpPaidMostOfHomeUpkeep`]: createBooleanWrapper(true),
      [`/hasIpPin`]: createBooleanWrapper(false),
    }),
    [`/familyAndHousehold`]: createCollectionWrapper([dependentId]),
    [`/hasCdccCarryoverAmountFromPriorTaxYear`]: createBooleanWrapper(false),
    [`/writableHasCdccForfeitedCredits`]: createBooleanWrapper(false),
  };

  describe(`that is employer furnished`, () => {
    const employerFurnishedData = {
      ...baseData,
      [`/writablePrimaryFilerDependentCarePlanMaximum`]: createDollarWrapper(`1000`),
      [`/writableCdccTotalQualifiedDependentCareExpenses`]: createDollarWrapper(`1000`),
      [Path.concretePath(`/cdccCareProviders`, providerId)]: createCollectionWrapper([providerId]),
    };

    describe(`when filer has only one employer`, () => {
      const singleEmployerData = {
        ...employerFurnishedData,
        ...makeW2Data(50_000, w2Id, primaryFilerId),
        [`/formW2s`]: createCollectionWrapper([w2Id]),
        [Path.concretePath(`/formW2s/*/writableDependentCareBenefits`, w2Id)]: createDollarWrapper(`1000`),
      };

      it(`is only complete when all required questions are answered and ../hasW2Employer is true`, () => {
        const { factGraph } = setupFactGraph(singleEmployerData);

        const getIsComplete = () => factGraph.get(Path.concretePath(`/cdccCareProviders/*/isComplete`, providerId));

        const requiredFacts = [
          [`/cdccCareProviders/*/writableIsOrganization`, true],
          [`/cdccCareProviders/*/isEmployerFurnished`, true],
          [`/cdccCareProviders/*/hasW2Employer`, true],
        ] satisfies [AbsolutePath & `/cdccCareProviders/*/${string}`, string | boolean][];

        for (const [factPath, factValue] of requiredFacts) {
          // Care Provider should not be considered complete before we set this fact
          expect(getIsComplete().get).toBe(false);

          factGraph.set(Path.concretePath(factPath, providerId), factValue);
          factGraph.save();
        }

        expect(getIsComplete().get).toBe(true);
      });

      it(`is NOT complete when all required questions are answered and ../hasW2Employer is false`, () => {
        const { factGraph } = setupFactGraph({
          ...singleEmployerData,
          ...makeCollectionItem(providerId, `/cdccCareProviders`, {
            [`/writableIsOrganization`]: createBooleanWrapper(true),
            [`/isEmployerFurnished`]: createBooleanWrapper(true),

            [`/hasW2Employer`]: createBooleanWrapper(false),
          }),
        });

        const isCompleteFact = factGraph.get(Path.concretePath(`/cdccCareProviders/*/isComplete`, providerId));
        expect(isCompleteFact.get).toBe(false);
      });
    });

    describe(`when filer has multiple employers`, () => {
      const secondW2Id = crypto.randomUUID();

      const multipleEmployersData = {
        ...employerFurnishedData,
        ...makeW2Data(40_000, w2Id, primaryFilerId),
        ...makeW2Data(10_000, secondW2Id, primaryFilerId),
        [`/formW2s`]: createCollectionWrapper([w2Id, secondW2Id]),
        [Path.concretePath(`/formW2s/*/writableDependentCareBenefits`, w2Id)]: createDollarWrapper(`1000`),
        [Path.concretePath(`/formW2s/*/writableDependentCareBenefits`, secondW2Id)]: createDollarWrapper(`1000`),
      };

      it(`is only complete when all required questions are answered`, () => {
        const { factGraph } = setupFactGraph(multipleEmployersData);

        const maybeEligibleForDependentCareBenefitsFact = factGraph.get(
          Path.concretePath(`/maybeEligibleForDependentCareBenefits`, null)
        );
        expect(maybeEligibleForDependentCareBenefitsFact.get).toBe(true);

        const getIsComplete = () => factGraph.get(Path.concretePath(`/cdccCareProviders/*/isComplete`, providerId));

        const requiredFacts = [
          [`/cdccCareProviders/*/writableIsOrganization`, true],
          [`/cdccCareProviders/*/isEmployerFurnished`, true],
          [`/cdccCareProviders/*/hasW2Employer`, true],
          [
            `/cdccCareProviders/*/writableEmployerWhoFurnishedCare`,
            CollectionItemReferenceFactory(w2Id, `/formW2s`, factGraph.sfgGraph).right,
          ],
        ] satisfies [AbsolutePath & `/cdccCareProviders/*/${string}`, _: unknown][];

        for (const [factPath, factValue] of requiredFacts) {
          // Care Provider should not be considered complete before we set this fact
          expect(getIsComplete().get).toBe(false);

          factGraph.set(Path.concretePath(factPath, providerId), factValue);
          factGraph.save();
        }

        expect(getIsComplete().get).toBe(true);
      });

      it(`is NOT complete when all required questions are answered and ../hasW2Employer is false`, () => {
        const { factGraph } = setupFactGraph({
          ...multipleEmployersData,
          ...makeCollectionItem(providerId, `/cdccCareProviders`, {
            [`/writableIsOrganization`]: createBooleanWrapper(true),
            [`/isEmployerFurnished`]: createBooleanWrapper(true),

            [`/hasW2Employer`]: createBooleanWrapper(false),
            [`/writableEmployerWhoFurnishedCare`]: createCollectionItemWrapper(w2Id),
          }),
        });

        const isCompleteFact = factGraph.get(Path.concretePath(`/cdccCareProviders/*/isComplete`, providerId));
        expect(isCompleteFact.get).toBe(false);
      });
    });
  });

  describe(`that is not employer furnished`, () => {
    const organizationProviderData = {
      ...baseData,
      ...makeW2Data(50_000, w2Id, primaryFilerId),
      [`/formW2s`]: createCollectionWrapper([w2Id]),
      ...makeCollectionItem(providerId, `/cdccCareProviders`, {
        [`/writableIsOrganization`]: createBooleanWrapper(true),
        [`/isEmployerFurnished`]: createBooleanWrapper(false),
        [`/writableOrganizationName`]: createStringWrapper(`Foo Bar Inc`),
        [`/writableAddress`]: createAddressWrapper(),
      }),
      [`/cdccCareProviders`]: createCollectionWrapper([providerId]),
    };

    describe(`that is a tax-exempt organization`, () => {
      it(`is only complete when all required questions are complete`, () => {
        const { factGraph } = setupFactGraph(organizationProviderData);

        const getIsComplete = () => factGraph.get(Path.concretePath(`/cdccCareProviders/*/isComplete`, providerId));

        const requiredFacts = [
          [`/cdccCareProviders/*/writableIsTaxExempt`, true],
          [`/cdccCareProviders/*/writableAmountPaidForCare`, DollarFactory(`1000`).right],
        ] satisfies [AbsolutePath & `/cdccCareProviders/*/${string}`, _: unknown][];

        for (const [factPath, factValue] of requiredFacts) {
          // Care Provider should not be considered complete before we set this fact
          expect(getIsComplete().get).toBe(false);

          factGraph.set(Path.concretePath(factPath, providerId), factValue);
          factGraph.save();
        }

        expect(getIsComplete().get).toBe(true);
      });
    });

    describe(`that is NOT a tax-exempt organization`, () => {
      describe(`and the filer has the providers EIN`, () => {
        it(`is only complete when all required questions are complete`, () => {
          const { factGraph } = setupFactGraph(organizationProviderData);

          const getIsComplete = () => factGraph.get(Path.concretePath(`/cdccCareProviders/*/isComplete`, providerId));

          const requiredFacts = [
            [`/cdccCareProviders/*/writableIsTaxExempt`, false],
            [`/cdccCareProviders/*/hasTinOrEin`, true],
            [`/cdccCareProviders/*/writableEin`, EinFactory(`009999999`).right],
            [`/cdccCareProviders/*/writableAmountPaidForCare`, DollarFactory(`1000`).right],
          ] satisfies [AbsolutePath & `/cdccCareProviders/*/${string}`, _: unknown][];

          for (const [factPath, factValue] of requiredFacts) {
            // Care Provider should not be considered complete before we set this fact
            expect(getIsComplete().get).toBe(false);

            factGraph.set(Path.concretePath(factPath, providerId), factValue);
            factGraph.save();
          }

          expect(getIsComplete().get).toBe(true);
        });
      });

      describe(`and the filer does not have the providers EIN`, () => {
        it(`is only complete when all required questions are complete`, () => {
          const { factGraph } = setupFactGraph(organizationProviderData);

          const getIsComplete = () => factGraph.get(Path.concretePath(`/cdccCareProviders/*/isComplete`, providerId));

          const requiredFacts = [
            [`/cdccCareProviders/*/writableIsTaxExempt`, false],
            [`/cdccCareProviders/*/hasTinOrEin`, false],
            [
              `/cdccCareProviders/*/writableDueDiligence`,
              MultiEnumFactory(
                jsSetToScalaSet(new Set(`providerMovedAndFilerUnableToFindThem`)),
                `/cdccDueDiligenceOptions`
              ).right,
            ],
            [`/cdccCareProviders/*/writableAmountPaidForCare`, DollarFactory(`1000`).right],
          ] satisfies [AbsolutePath & `/cdccCareProviders/*/${string}`, _: unknown][];

          for (const [factPath, factValue] of requiredFacts) {
            // Care Provider should not be considered complete before we set this fact
            expect(getIsComplete().get).toBe(false);

            factGraph.set(Path.concretePath(factPath, providerId), factValue);
            factGraph.save();
          }

          expect(getIsComplete().get).toBe(true);
        });
      });
    });

    describe(`that is an individual`, () => {
      const individualProviderData = {
        ...baseData,
        ...makeW2Data(50_000, w2Id, primaryFilerId),
        [`/formW2s`]: createCollectionWrapper([w2Id]),
        ...makeCollectionItem(providerId, `/cdccCareProviders`, {
          [`/writableIsOrganization`]: createBooleanWrapper(false),
        }),
        [`/cdccCareProviders`]: createCollectionWrapper([providerId]),
      };
      describe(`and the filer has the provider's SSN`, () => {
        it(`is only complete when all required questions are complete`, () => {
          const { factGraph } = setupFactGraph(individualProviderData);

          const getIsComplete = () => factGraph.get(Path.concretePath(`/cdccCareProviders/*/isComplete`, providerId));

          const requiredFacts = [
            [`/cdccCareProviders/*/writableFirstName`, `Flib`],
            [`/cdccCareProviders/*/writableLastName`, `McJib`],
            [`/cdccCareProviders/*/writableAddress`, AddressFactory(`123 Main Street`, `City`, `32013`, `FL`).right],
            [`/cdccCareProviders/*/hasTinOrEin`, true],
            [`/cdccCareProviders/*/writableTin`, TinFactory(`555005555`).right],
            [`/cdccCareProviders/*/writableAmountPaidForCare`, DollarFactory(`1000`).right],
          ];

          for (const [factPath, factValue] of requiredFacts) {
            // Care Provider should not be considered complete before we set this fact
            expect(getIsComplete().get).toBe(false);

            factGraph.set(Path.concretePath(factPath, providerId), factValue);
            factGraph.save();
          }

          expect(getIsComplete().get).toBe(true);
        });
      });
      describe(`and the filer does not have the providers SSN`, () => {
        it(`is only complete when all required questions are complete`, () => {
          const { factGraph } = setupFactGraph(individualProviderData);

          const getIsComplete = () => factGraph.get(Path.concretePath(`/cdccCareProviders/*/isComplete`, providerId));

          const requiredFacts = [
            [`/cdccCareProviders/*/writableFirstName`, `Flib`],
            [`/cdccCareProviders/*/writableLastName`, `McJib`],
            [`/cdccCareProviders/*/writableAddress`, AddressFactory(`123 Main Street`, `City`, `32013`, `FL`).right],
            [`/cdccCareProviders/*/hasTinOrEin`, false],
            [
              `/cdccCareProviders/*/writableDueDiligence`,
              MultiEnumFactory(
                jsSetToScalaSet(new Set(`providerMovedAndFilerUnableToFindThem`)),
                `/cdccDueDiligenceOptions`
              ).right,
            ],
            [`/cdccCareProviders/*/writableAmountPaidForCare`, DollarFactory(`1000`).right],
          ] satisfies [AbsolutePath & `/cdccCareProviders/*/${string}`, _: unknown][];

          for (const [factPath, factValue] of requiredFacts) {
            // Care Provider should not be considered complete before we set this fact
            expect(getIsComplete().get).toBe(false);

            factGraph.set(Path.concretePath(factPath, providerId), factValue);
            factGraph.save();
          }

          expect(getIsComplete().get).toBe(true);
        });
      });
    });
  });
});

/**
 * These tests check the flowKnockoutBenefitsDistribution fact.
 * It is a boolean fact that is true if the filer should be knocked out due to unsupported calculations
 * being necessary to maximize their CDC credit.
 */

// TODO! This knockout should also be considering TY-1 benefits.
// What matters is that some of the dependent care benefits are taxable
describe(`For /flowKnockoutBenefitsDistribution, with filers`, () => {
  const primaryFilerId = uuid;
  const spouseId = `859c03d1-af4a-447f-96aa-d19397048a48`;
  const tpW2Id = `79cb9b1b-7898-429a-8681-2dd64c06a962`;
  const spouseW2Id = `3d56a54d-09cc-4f9f-abd5-6dd48508bf95`;
  const dependentId = `85501abd-9f7c-4746-b4e9-f765565ca923`;

  const qualifyingDependent = {
    ...makeChildData(dependentId, `${CURRENT_TAX_YEAR_AS_NUMBER - 8}-01-01`),
    [`/familyAndHousehold`]: createCollectionWrapper([dependentId]),
    [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(false),
  };

  const initialData = {
    ...sharedData,
    ...makeFilerData(primaryFilerId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
    ...makeFilerData(spouseId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
    ...livedTogetherFacts,
    [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
    [`/filingStatus`]: createEnumWrapper(FilingStatus.MFJ, `/filingStatusOptions`),

    ...makeW2Data(2000, tpW2Id),
    ...makeW2Data(2000, spouseW2Id),
    '/formW2s': createCollectionWrapper([tpW2Id, spouseW2Id]),
    [`/formW2s/#${tpW2Id}/filer`]: createCollectionItemWrapper(primaryFilerId),
    [`/formW2s/#${spouseW2Id}/filer`]: createCollectionItemWrapper(spouseId),
    [`/formW2s/#${tpW2Id}/writableDependentCareBenefits`]: createDollarWrapper(`3000.00`),
    [`/formW2s/#${spouseW2Id}/writableDependentCareBenefits`]: createDollarWrapper(`3000.00`),
    '/writablePrimaryFilerDependentCarePlanMaximum': createDollarWrapper(`3000.00`),
    '/writableSecondaryFilerDependentCarePlanMaximum': createDollarWrapper(`3000.00`),

    [Path.concretePath(`/filers/*/isDisabled`, primaryFilerId)]: createBooleanWrapper(false),
    [Path.concretePath(`/filers/*/isDisabled`, spouseId)]: createBooleanWrapper(false),

    ...qualifyingDependent,
    [Path.concretePath(`/familyAndHousehold/*/cdccHasDependentCareExpenses`, dependentId)]: createBooleanWrapper(true),
    [Path.concretePath(`/familyAndHousehold/*/cdccHadExpensesPaidToQualifyingProvider`, dependentId)]:
      createBooleanWrapper(true),
    [`/familyAndHousehold/#${dependentId}/writableCdccQualifyingExpenseAmount`]: createDollarWrapper(`6000.00`),

    [`/writableCdccTotalQualifiedDependentCareExpenses`]: createDollarWrapper(`6000.00`),
    [Path.concretePath(`/hasCdccCarryoverAmountFromPriorTaxYear`, null)]: createBooleanWrapper(false),
    [Path.concretePath(`/writableHasCdccForfeitedCredits`, null)]: createBooleanWrapper(false),
  };

  describe(`Where both received dependent care benefits`, () => {
    describe(`And both had earned income below the benefit cap`, () => {
      const w2sBelowBenefitCap = {
        ...initialData,
        [`/formW2s/#${tpW2Id}/writableWages`]: createDollarWrapper(`1000`),
        [`/formW2s/#${spouseW2Id}/writableWages`]: createDollarWrapper(`1000`),
      };

      it(`will see the benefits distribution knockout flow`, () => {
        const { factGraph } = setupFactGraph(w2sBelowBenefitCap);
        const fact = factGraph.get(Path.concretePath(`/flowKnockoutBenefitsDistribution`, null));
        expect(fact.get).toBe(true);
      });
    });

    describe(`Only one had earned income below the benefit cap`, () => {
      const w2BelowBenefitCap = {
        ...initialData,
        [`/formW2s/#${tpW2Id}/writableWages`]: createDollarWrapper(`1000`),
        [`/formW2s/#${spouseW2Id}/writableWages`]: createDollarWrapper(`4000`),
      };
      it(`will see the benefits distribution knockout flow`, () => {
        const { factGraph } = setupFactGraph(w2BelowBenefitCap);
        const fact = factGraph.get(Path.concretePath(`/flowKnockoutBenefitsDistribution`, null));
        expect(fact.get).toBe(false);
      });
    });

    describe(`Neither had earned income below the benefit cap`, () => {
      const w2sAboveBenefitCap = {
        ...initialData,
        [`/formW2s/#${tpW2Id}/writableWages`]: createDollarWrapper(`4000`),
        [`/formW2s/#${spouseW2Id}/writableWages`]: createDollarWrapper(`4000`),
      };

      it(`will not see the benefits distribution knockout flow`, () => {
        const { factGraph } = setupFactGraph(w2sAboveBenefitCap);
        const fact = factGraph.get(Path.concretePath(`/flowKnockoutBenefitsDistribution`, null));
        expect(fact.get).toBe(false);
      });
    });

    describe(`Filers with only carryover dependent benefits`, () => {
      const largerW2s = {
        ...initialData,
        [`/formW2s/#${tpW2Id}/writableWages`]: createDollarWrapper(`2000`),
        [`/formW2s/#${spouseW2Id}/writableWages`]: createDollarWrapper(`2000`),
        [`/formW2s/#${tpW2Id}/writableDependentCareBenefits`]: createDollarWrapper(`0.00`),
        [`/formW2s/#${spouseW2Id}/writableDependentCareBenefits`]: createDollarWrapper(`1000.00`),

        [Path.concretePath(`/hasCdccCarryoverAmountFromPriorTaxYear`, null)]: createBooleanWrapper(true),
        [Path.concretePath(`/writableCdccCarryoverAmountFromPriorTaxYear`, null)]: createDollarWrapper(`3000`),
        [`/writableCdccTotalQualifiedDependentCareExpenses`]: createDollarWrapper(`4000.00`),
      };

      it(`will not see the benefits distribution knockout flow`, () => {
        const { factGraph } = setupFactGraph(largerW2s);
        const fact = factGraph.get(Path.concretePath(`/flowKnockoutBenefitsDistribution`, null));
        expect(fact.get).toBe(false);
      });
    });

    describe(`One filer had much more earned income than the other`, () => {
      // One filer made more than the other filer's earned income + reported dependent care benefits
      const mixMatchW2s = {
        ...initialData,
        [`/formW2s/#${tpW2Id}/writableWages`]: createDollarWrapper(`4000`),
        [`/formW2s/#${spouseW2Id}/writableWages`]: createDollarWrapper(`7000`),
      };

      it(`will not see the benefits distribution knockout flow`, () => {
        const { factGraph } = setupFactGraph(mixMatchW2s);
        const fact = factGraph.get(Path.concretePath(`/flowKnockoutBenefitsDistribution`, null));
        expect(fact.get).toBe(false);
      });
    });

    describe(`Filers with income greater than their expenses`, () => {
      const largerW2s = {
        ...initialData,
        [`/formW2s/#${tpW2Id}/writableWages`]: createDollarWrapper(`30000`),
        [`/formW2s/#${spouseW2Id}/writableWages`]: createDollarWrapper(`35000`),
      };

      it(`will not see the benefits distribution knockout flow`, () => {
        const { factGraph } = setupFactGraph(largerW2s);
        const fact = factGraph.get(Path.concretePath(`/flowKnockoutBenefitsDistribution`, null));
        expect(fact.get).toBe(false);
      });
    });
  });
});

describe(`A family and household member who is a qualifying child dependent`, () => {
  const tpW2Id = `54fb6ba0-aeae-40c0-8393-3a66468cdfcc`;
  const singleFiler = {
    ...sharedData,
    ...makeFilerData(primaryFilerId, { [`/canBeClaimed`]: createBooleanWrapper(false) }),
    [`/maritalStatus`]: createEnumWrapper(`neverMarried`, `/maritalStatusOptions`),
    [`/filingStatus`]: createEnumWrapper(FilingStatus.SINGLE, `/filingStatusOptions`),
    ...makeW2Data(50000, tpW2Id),
    '/formW2s': {
      $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
      item: { items: [tpW2Id] },
    },
    [`/formW2s/#${tpW2Id}/filer`]: {
      item: {
        id: primaryFilerId,
      },
      $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
    },
  };
  const under13Dob = `${CURRENT_TAX_YEAR_AS_NUMBER - 8}-01-01`;
  const turned13Dob = `${CURRENT_TAX_YEAR_AS_NUMBER - 13}-02-01`;
  const turned13Jan1Dob = `${CURRENT_TAX_YEAR_AS_NUMBER - 13}-01-01`;
  const dependentId = `83c353fb-e4cf-4b98-aacd-49619ebaac2f`;

  const makeCdccQualifyingDependent = (dob: string) => {
    return {
      ...makeChildData(dependentId, dob),
      [`/familyAndHousehold`]: createCollectionWrapper([dependentId]),
      [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(false),
      [Path.concretePath(`/familyAndHousehold/*/unableToCareForSelf`, dependentId)]: createBooleanWrapper(false),
    };
  };

  const cases = [
    { msg: `who was under age 13 the whole tax year is a cdcc qualifying child`, dob: under13Dob, qualifies: true },
    { msg: `who turned 13 after Jan 1 is a cdcc qualifying child`, dob: turned13Dob, qualifies: true },
    {
      msg: `who turned 13 in the tax year on Jan 1 is NOT a cdcc qualifying child`,
      dob: turned13Jan1Dob,
      qualifies: false,
    },
  ];

  it.each(cases)(`%s`, (scenario) => {
    const data = {
      ...singleFiler,
      ...makeCdccQualifyingDependent(scenario.dob),
    };

    const { factGraph } = setupFactGraph(data);

    const qualifyingPerson = factGraph.get(
      Path.concretePath(`/familyAndHousehold/*/cdccQualifyingPerson`, dependentId)
    );
    expect(qualifyingPerson.get).toBe(scenario.qualifies);
  });
});
