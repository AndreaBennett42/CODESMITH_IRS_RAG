import { it, describe, expect } from 'vitest';
import { baseFilerData, primaryFilerId, baseDependentId, makeW2Data } from '../testData.js';
import {
  createBooleanWrapper,
  createDayWrapper,
  createEnumWrapper,
  createStringWrapper,
} from '../persistenceWrappers.js';
import { createFlowConfig } from '../../flow/flowConfig.js';
import flowNodes from '../../flow/flow.js';
import makeGivenFacts from './functionalFlowUtils.js';
import { CURRENT_TAX_YEAR, TAX_YEAR_2023 } from '../../constants/taxConstants.js';
import { setupFactGraphDeprecated } from '../setupFactGraph.js';

const THRESHOLDS = TAX_YEAR_2023.EITC_INCOME_THRESHOLDS;

const flow = createFlowConfig(flowNodes);
const givenFacts = makeGivenFacts(flow);

describe(`The \`dependents\` subcategory`, () => {
  const dependentId = baseDependentId;
  const baseDependentData = {
    ...baseFilerData,
    '/familyAndHousehold': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [dependentId] } },
    [`/familyAndHousehold/#${dependentId}/firstName`]: createStringWrapper(`John`),
    [`/familyAndHousehold/#${dependentId}/middleInitial`]: createStringWrapper(`Q`),
    [`/familyAndHousehold/#${dependentId}/lastName`]: createStringWrapper(`Dependent`),
  };

  // Boundary test values
  const birthYearIfTurnedXInTaxYear = (x: number): number => parseInt(CURRENT_TAX_YEAR) - x;
  const dob = {
    barelyTwentyAtEndOfYear: `${birthYearIfTurnedXInTaxYear(20)}-12-31`,
    justShortOfTwentyAtEndOfYear: `${birthYearIfTurnedXInTaxYear(19)}-01-01`,

    barelyFourteenAtStartOfYear: `${birthYearIfTurnedXInTaxYear(14)}-01-01`,
    justShortOfFourteenAtStartOfYear: `${birthYearIfTurnedXInTaxYear(14)}-01-02`,

    barelyNineteenAtEndOfYear: `${birthYearIfTurnedXInTaxYear(19)}-12-31`,

    eigtheenOnJulyThird: `${birthYearIfTurnedXInTaxYear(18)}-07-03`,
    eighteenOnJulySecond: `${birthYearIfTurnedXInTaxYear(18)}-07-02`,

    justShortOfTwentyOneOnJulySecond: `${birthYearIfTurnedXInTaxYear(21)}-07-03`,
    barelyTwentyOneOnJulySecond: `${birthYearIfTurnedXInTaxYear(21)}-07-02`,
  };

  const dobJustShortOf18 = dob.eigtheenOnJulyThird;

  // Screen paths
  const screen = {
    basicInfo: `/flow/you-and-your-family/dependents/add-person-basic-info`,
    relationshipCategory: `/flow/you-and-your-family/dependents/add-person-relationship-category`,
    relationshipTypeChild: `/flow/you-and-your-family/dependents/add-person-relationship-type-child`,
    residencyDuration: `/flow/you-and-your-family/dependents/add-person-lived-with-you`,
    selfCare: `/flow/you-and-your-family/dependents/add-person-self-care`,
    tin: `/flow/you-and-your-family/dependents/add-person-tin`,
    livedWithTpInUs: `/flow/you-and-your-family/dependents/add-person-lived-with-you-in-us`,
    isThereAnotherParent: `/flow/you-and-your-family/dependents/add-person-another-parent-in-picture`,
    notInCustodyOtherRel: `/flow/you-and-your-family/dependents/add-person-not-in-custody-other-rel`,
    areParentsLiving: `/flow/you-and-your-family/dependents/add-person-parents-living`,
    parentNotClaiming: `/flow/you-and-your-family/dependents/add-person-any-parents-written-declaration`,
    otherParentNotClaiming: `/flow/you-and-your-family/dependents/add-person-other-parent-written-declaration`,
    writtenDeclaredSigned: `/flow/you-and-your-family/dependents/add-person-written-declaration-signed`,
    notInCustody: `/flow/you-and-your-family/dependents/add-person-not-in-custody`,
    // eslint-disable-next-line max-len
    livedWithTpOrOtherParentMoreThanSixMonths: `/flow/you-and-your-family/dependents/add-person-special-rule-residency`,
    nightsWithTpVsOtherParent: `/flow/you-and-your-family/dependents/add-person-special-rule-nights-together`,
    agiTiebreaker: `/flow/you-and-your-family/dependents/add-person-special-rule-agi-tiebreaker`,
    parentalSituation: `/flow/you-and-your-family/dependents/add-person-special-rule-living-marital-status`,
    parentalSupport: `/flow/you-and-your-family/dependents/add-person-special-rule-support`,
    custodialParentAck: `/flow/you-and-your-family/dependents/add-person-special-rule-applies-custodial`,
    noncustodialParentAck: `/flow/you-and-your-family/dependents/add-person-special-rule-applies-non-custodial`,
    conflictingInfoKnockout: `/flow/you-and-your-family/dependents/add-person-special-rule-conflicting-information`,
    specialRuleDoesNotApply: `/flow/you-and-your-family/dependents/add-person-special-rule-does-not-apply`,
    keepingUpHome: `/flow/you-and-your-family/dependents/add-person-keeping-up-home`,
    keepingUpParentsHome: `/flow/you-and-your-family/dependents/add-person-keeping-up-parents-home`,
    ssnValidForWork: `/flow/you-and-your-family/dependents/add-person-ssn-valid-for-work`,
    acknowledgeTin: `/flow/you-and-your-family/dependents/add-person-acknowledge-tin`,
    qcSupportTest: `/flow/you-and-your-family/dependents/add-person-qc-support`,
    qrSupportTest: `/flow/you-and-your-family/dependents/add-person-qr-support`,
    qcOfAnother: `/flow/you-and-your-family/dependents/add-person-qc-of-another`,
    married: `/flow/you-and-your-family/dependents/add-person-married`,
    jointReturn: `/flow/you-and-your-family/dependents/add-person-joint-return`,
    requiredToFile: `/flow/you-and-your-family/dependents/add-person-filing-requirement`,
    filingForRefundOnly: `/flow/you-and-your-family/dependents/add-person-filing-for-refund-only`,
    citizenship: `/flow/you-and-your-family/dependents/add-person-us-citizen`,
    // eslint-disable-next-line max-len
    qcOfAnotherClaimerFilingRequirement: `/flow/you-and-your-family/dependents/add-person-qc-of-another-claimer-filing-requirement`,
    qcOfAnotherClaimerFiling: `/flow/you-and-your-family/dependents/add-person-qc-of-another-claimer-filing`,
    qcOfAnotherClaimersReturn: `/flow/you-and-your-family/dependents/add-person-qc-of-another-claimers-return`,
    qualifiedQCOfMultipleTps: `/flow/you-and-your-family/dependents/qualified-qc-of-multiple-tps`,
    potentialQP: `/flow/you-and-your-family/dependents/potential-qp`,
    confirmedQP: `/flow/you-and-your-family/dependents/confirmed-qp`,
    qcOfAnotherSummary: `/flow/you-and-your-family/dependents/add-person-qc-of-another-summary`,
    notQualifiedForAny: `/flow/you-and-your-family/dependents/not-qualified`,
    qualifiedDependent: `/flow/you-and-your-family/dependents/qualified-dependent`,
    dataView: `/data-view/flow/you-and-your-family/dependents`,
  };

  // TODO pull these from flow.xml ? If these become out of sync, test results cannot be trusted.
  const enums = {
    residencyDuration: [`allYear`, `sixToElevenMonths`, `lessThanSixMonths`],
    relationship: [
      `biologicalChild`,
      `adoptedChild`,
      `stepChild`,
      `fosterChild`,
      `grandChild`,
      `otherDescendantOfChild`,
      `childInLaw`,
      `sibling`,
      `childOfSibling`,
      `halfSibling`,
      `childOfHalfSibling`,
      `stepSibling`,
      `childOfStepSibling`,
      `otherDescendantOfSibling`,
      `siblingInLaw`,
      `parent`,
      `siblingOfParent`,
      `grandParent`,
      `otherAncestorOfParent`,
      `stepParent`,
      `parentInLaw`,
      `noneOfTheAbove`,
    ],
    moreLessEqual: [`more`, `less`, `equal`],
    filingStatus: [
      `marriedFilingJointly`,
      `qualifiedSurvivingSpouse`,
      `headOfHousehold`,
      `single`,
      `marriedFilingSeparately`,
    ],
    parentalSituation: [`divorcedOrSeparated`, `livedApartLastSixMonths`, `other`],
    maritalStatus: [`single`, `married`, `divorced`, `widowed`],
    yearOfSpouseDeath: [`taxYear`, `taxYearMinusOne`, `taxYearMinusTwo`, `beforeTaxYearMinusTwo`],
    tinType: [`ssn`, `itin`, `atin`, `none`],
    ssnEmploymentValidity: [
      `notValid`,
      `validOnlyWithDhsAuthorizationInEffect`,
      `validOnlyWithDhsAuthorizationExpired`,
      `neither`,
    ],
  };

  describe(`The "Student, Disability" segment`, () => {
    describe(`When the dependent is barely 20 at end of year`, () => {
      const { factGraph } = setupFactGraphDeprecated({
        ...baseDependentData,
        [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyTwentyAtEndOfYear),
      });
      it(`asks about student status ...`, ({ task }) => {
        expect(givenFacts(factGraph).atPath(screen.relationshipTypeChild, dependentId, task)).toRouteNextTo(
          `/flow/you-and-your-family/dependents/add-person-student`
        );
      });
      it(`... and then about disability ...`, ({ task }) => {
        expect(
          givenFacts(factGraph).atPath(`/flow/you-and-your-family/dependents/add-person-student`, dependentId, task)
        ).toRouteNextTo(`/flow/you-and-your-family/dependents/add-person-disability`);
      });
    });
    describe(`When the dependent is just short of 14 at start of year`, () => {
      const { factGraph } = setupFactGraphDeprecated({
        ...baseDependentData,
        [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.justShortOfFourteenAtStartOfYear),
      });
      it(`skips this entire segment`, ({ task }) => {
        expect(givenFacts(factGraph).atPath(screen.relationshipTypeChild, dependentId, task)).toRouteNextTo(
          screen.selfCare
        );
      });
    });
  });

  describe(`The "Residency duration" segment`, () => {
    describe(`When the filer has lived with the dependent all year`, () => {
      it(`asks about residency with filer in U.S.`, ({ task }) => {
        const { factGraph } = setupFactGraphDeprecated({
          ...baseDependentData,
          [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
            `allYear`,
            `/residencyDurationOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyFourteenAtStartOfYear),
          [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
            `childOrDescendants`,
            `/relationshipCategoryOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
            `biologicalChild`,
            `/childRelationshipOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/hasOtherBiologicalOrAdoptiveParent`]: createBooleanWrapper(false),
        });
        expect(givenFacts(factGraph).atPath(screen.isThereAnotherParent, dependentId, task)).toRouteNextTo(
          screen.livedWithTpInUs
        );
      });
    });

    describe(`When the filer has lived with the dependent six to eleven months`, () => {
      it(`asks about residency with filer in U.S.`, ({ task }) => {
        const { factGraph } = setupFactGraphDeprecated({
          ...baseDependentData,
          [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
            `sixToElevenMonths`,
            `/residencyDurationOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyFourteenAtStartOfYear),
          [`/familyAndHousehold/#${dependentId}/hasOtherBiologicalOrAdoptiveParent`]: createBooleanWrapper(false),
          [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
            `childOrDescendants`,
            `/relationshipCategoryOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
            `biologicalChild`,
            `/childRelationshipOptions`
          ),
        });
        expect(givenFacts(factGraph).atPath(screen.isThereAnotherParent, dependentId, task)).toRouteNextTo(
          screen.livedWithTpInUs
        );
      });
    });

    describe(`When the TP would be ineligible for the EITC and HoH/QSS`, () => {
      it(`asks about the TIN, skipping residency duration`, ({ task }) => {
        const { factGraph } = setupFactGraphDeprecated({
          ...baseDependentData,
          // Must be a qualifying child to create HoH eligibility and not have residence for > 6 months
          [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
            `childOrDescendants`,
            `/relationshipCategoryOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
            `biologicalChild`,
            `/childRelationshipOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
            `lessThanSixMonths`,
            `/residencyDurationOptions`
          ),
        });
        expect(givenFacts(factGraph).atPath(screen.residencyDuration, dependentId, task)).toRouteNextTo(screen.tin);
      });
    });
  });

  describe(`The "Special Rule Benefit Split" segment`, () => {
    // This segment is skipped unless all of three conditions are met.
    // Each of the following three blocks isolate one of those three conditions.
    // The exit segment is tin, and not cost of keeping up home because the
    // filer does not qualify for HoH.
    const exitSegment = screen.tin;

    describe(`when the dependent fails the QC age test`, () => {
      const { factGraph } = setupFactGraphDeprecated({
        ...baseDependentData,
        // We set this residency duration so that the exit segment will skip the
        // lived with TP in the US question
        [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
          `lessThanSixMonths`,
          `/residencyDurationOptions`
        ),
        // fail condition 1: must pass age test
        // meet condition 2: must be under 21 on July 2
        [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyNineteenAtEndOfYear),
        // meet condition 3: must pass relationship test
        [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
          `childOrDescendants`,
          `/relationshipCategoryOptions`
        ),
        [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
          `biologicalChild`,
          `/childRelationshipOptions`
        ),
      });
      it(`skips this segment completely`, ({ task }) => {
        expect(givenFacts(factGraph).atPath(screen.residencyDuration, dependentId, task)).toRouteNextTo(exitSegment);
      });
    });

    describe(`when the dependent is 21 or older on July 2`, () => {
      const { factGraph } = setupFactGraphDeprecated({
        ...baseDependentData,
        // We set this residency duration so that the exit segment will skip the
        // lived with TP in the US question
        [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
          `lessThanSixMonths`,
          `/residencyDurationOptions`
        ),
        // meet condition 1: must pass age test
        // fail condition 2: must be under 21 on July 2
        [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyTwentyOneOnJulySecond),
        // meet condition 3: must pass relationship test
        [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
          `childOrDescendants`,
          `/relationshipCategoryOptions`
        ),
        [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
          `biologicalChild`,
          `/childRelationshipOptions`
        ),
      });
      it(`skips this segment completely`, ({ task }) => {
        expect(givenFacts(factGraph).atPath(screen.residencyDuration, dependentId, task)).toRouteNextTo(exitSegment);
      });
    });

    describe(`when the dependent fails the relationship test`, () => {
      const { factGraph } = setupFactGraphDeprecated({
        ...baseDependentData,
        // meet condition 1: must pass age test
        // meet condition 2: must be under 21 on July 2
        [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyFourteenAtStartOfYear),
        // fail condition 3: must pass relationship test
        [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
          `parentOrAncestors`,
          `/relationshipCategoryOptions`
        ),
        [`/familyAndHousehold/#${dependentId}/parentalRelationship`]: createEnumWrapper(
          `parent`,
          `/parentalRelationshipOptions`
        ),
      });
      it(`skips this segment completely`, ({ task }) => {
        expect(givenFacts(factGraph).atPath(screen.residencyDuration, dependentId, task)).toRouteNextTo(exitSegment);
      });
    });

    // There are two entry points to this segment. The next two blocks test their conditions.
    describe(`When dependent is filer's child ...`, () => {
      describe(`... by birth (biological)`, () => {
        testFlowWhenDependentIsFilersChild({
          [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
            `childOrDescendants`,
            `/relationshipCategoryOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
            `biologicalChild`,
            `/childRelationshipOptions`
          ),
        });
      });

      describe(`... by adoption`, () => {
        testFlowWhenDependentIsFilersChild({
          [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
            `childOrDescendants`,
            `/relationshipCategoryOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
            `adoptedChild`,
            `/childRelationshipOptions`
          ),
        });
      });

      const testFlowWhenDependentIsFilersChild = (graphChildSettings: object) => {
        const graphSettings = {
          ...baseDependentData,
          ...graphChildSettings,
          [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyFourteenAtStartOfYear),
          [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
            `lessThanSixMonths`,
            `/residencyDurationOptions`
          ),
        };
        // Most tests use the following
        const graphSettingsHasOtherParent = {
          ...graphSettings,
          [`/familyAndHousehold/#${dependentId}/hasOtherBiologicalOrAdoptiveParent`]: createBooleanWrapper(true),
        };

        it(`asks if there is another parent besides filer/spouse`, ({ task }) => {
          const { factGraph } = setupFactGraphDeprecated(graphSettings);
          expect(givenFacts(factGraph).atPath(screen.residencyDuration, dependentId, task)).toRouteNextTo(
            screen.isThereAnotherParent
          );
        });
        it(`if there is another parent, ask if they signed a declaration not to claim dependent`, ({ task }) => {
          const { factGraph } = setupFactGraphDeprecated(graphSettingsHasOtherParent);
          expect(givenFacts(factGraph).atPath(screen.isThereAnotherParent, dependentId, task)).toRouteNextTo(
            screen.otherParentNotClaiming
          );
        });
        it(`if there is not another parent, exit this segment`, ({ task }) => {
          const { factGraph } = setupFactGraphDeprecated({
            ...graphSettings,
            [`/familyAndHousehold/#${dependentId}/hasOtherBiologicalOrAdoptiveParent`]: createBooleanWrapper(false),
          });
          expect(givenFacts(factGraph).atPath(screen.isThereAnotherParent, dependentId, task)).toRouteNextTo(
            exitSegment
          );
        });
        it(`if a parent has signed declaration, show acknowledgement`, ({ task }) => {
          const { factGraph } = setupFactGraphDeprecated({
            ...graphSettingsHasOtherParent,
            [`/familyAndHousehold/#${dependentId}/whichParentNotClaiming`]: createEnumWrapper(
              `iDid`,
              `/writtenDeclarationOptions`
            ),
          });
          expect(givenFacts(factGraph).atPath(screen.otherParentNotClaiming, dependentId, task)).toRouteNextTo(
            screen.writtenDeclaredSigned
          );
        });
        it(`if a parent did not sign declaration, exit the benefit split`, ({ task }) => {
          const { factGraph } = setupFactGraphDeprecated({
            ...graphSettingsHasOtherParent,
            [`/familyAndHousehold/#${dependentId}/whichParentNotClaiming`]: createEnumWrapper(
              `nobodyDid`,
              `/writtenDeclarationOptions`
            ),
          });
          expect(givenFacts(factGraph).atPath(screen.otherParentNotClaiming, dependentId, task)).toRouteNextTo(
            screen.tin
          );
        });

        describe(`post acknowledgement screens scenarios`, () => {
          const exitSegment = screen.specialRuleDoesNotApply;
          const graphSettingsPostAck = {
            ...baseDependentData,
            ...graphChildSettings,
            [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
              `lessThanSixMonths`,
              `/residencyDurationOptions`
            ),
            [`/familyAndHousehold/#${dependentId}/hasOtherBiologicalOrAdoptiveParent`]: createBooleanWrapper(true),
            // This activates the benefits split segment for people up to age 24 *at end of year*,
            // allowing us to test scenarios with logic based on age at July 2.
            [`/familyAndHousehold/#${dependentId}/fullTimeStudent`]: createBooleanWrapper(true),
          };

          describe(`after ${screen.writtenDeclaredSigned}`, () => {
            describe(`with dependent age just short of 18`, () => {
              const { factGraph } = setupFactGraphDeprecated({
                ...graphSettingsPostAck,
                [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dobJustShortOf18),
                [`/familyAndHousehold/#${dependentId}/whichParentNotClaiming`]: createEnumWrapper(
                  `iDid`,
                  `/writtenDeclarationOptions`
                ),
              });

              it(`goes to ${screen.notInCustody}`, ({ task }) => {
                expect(givenFacts(factGraph).atPath(screen.writtenDeclaredSigned, dependentId, task)).toRouteNextTo(
                  screen.notInCustody
                );
              });
            });
            describe(`with dependent age 21 who has disability`, () => {
              const { factGraph } = setupFactGraphDeprecated({
                ...graphSettingsPostAck,
                [`/familyAndHousehold/#${dependentId}/whichParentNotClaiming`]: createEnumWrapper(
                  `theyDid`,
                  `/writtenDeclarationOptions`
                ),
                [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyTwentyOneOnJulySecond),
                [`/familyAndHousehold/#${dependentId}/permanentTotalDisability`]: createBooleanWrapper(true),
              });

              it(`asks about custody`, ({ task }) => {
                expect(givenFacts(factGraph).atPath(screen.writtenDeclaredSigned, dependentId, task)).toRouteNextTo(
                  screen.notInCustody
                );
              });
            });
          });

          const graphSettings = {
            ...graphSettingsPostAck,
            [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dobJustShortOf18),
            [`/familyAndHousehold/#${dependentId}/whichParentNotClaiming`]: createEnumWrapper(
              `iDid`,
              `/writtenDeclarationOptions`
            ),
          };

          it(`if child is still in custody ask about residency`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
            });

            expect(givenFacts(factGraph).atPath(screen.notInCustody, dependentId, task)).toRouteNextTo(
              screen.livedWithTpOrOtherParentMoreThanSixMonths
            );
          });
          it(`else (child is not still in custody) leave this segment`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(false),
            });

            expect(givenFacts(factGraph).atPath(screen.notInCustody, dependentId, task)).toRouteNextTo(exitSegment);
          });
          it(`if resided with some parent for 6 months, ask to compare number of nights`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                createBooleanWrapper(true),
            });

            expect(
              givenFacts(factGraph).atPath(screen.livedWithTpOrOtherParentMoreThanSixMonths, dependentId, task)
            ).toRouteNextTo(screen.nightsWithTpVsOtherParent);
          });
          it(`else exit this segment`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                createBooleanWrapper(false),
            });

            expect(
              givenFacts(factGraph).atPath(screen.livedWithTpOrOtherParentMoreThanSixMonths, dependentId, task)
            ).toRouteNextTo(exitSegment);
          });
          it(`if same number of nights with parents, ask to compare AGIs`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/nightsWithTpVsOtherParent`]: createEnumWrapper(
                `equal`,
                `/moreLessEqualOptions`
              ),
            });

            expect(givenFacts(factGraph).atPath(screen.nightsWithTpVsOtherParent, dependentId, task)).toRouteNextTo(
              screen.agiTiebreaker
            );
          });
          it(`else if more nights with filer, ask about marital situation`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/nightsWithTpVsOtherParent`]: createEnumWrapper(
                `more`,
                `/moreLessEqualOptions`
              ),
            });

            expect(givenFacts(factGraph).atPath(screen.nightsWithTpVsOtherParent, dependentId, task)).toRouteNextTo(
              screen.parentalSituation
            );
          });
          it(`else if fewer nights with filer, ask about marital situation`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/nightsWithTpVsOtherParent`]: createEnumWrapper(
                `less`,
                `/moreLessEqualOptions`
              ),
            });

            expect(givenFacts(factGraph).atPath(screen.nightsWithTpVsOtherParent, dependentId, task)).toRouteNextTo(
              screen.parentalSituation
            );
          });
          it(`when filer's AGI is higher than other parent, ask about marital situation`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/nightsWithTpVsOtherParent`]: createEnumWrapper(
                `equal`,
                `/moreLessEqualOptions`
              ),
              [`/familyAndHousehold/#${dependentId}/tpAgiHigherThanOtherParent`]: createBooleanWrapper(true),
            });

            expect(givenFacts(factGraph).atPath(screen.agiTiebreaker, dependentId, task)).toRouteNextTo(
              screen.parentalSituation
            );
          });
          it(`when filer's AGI is lower than other parent, ask about marital situation`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/nightsWithTpVsOtherParent`]: createEnumWrapper(
                `equal`,
                `/moreLessEqualOptions`
              ),
              [`/familyAndHousehold/#${dependentId}/tpAgiHigherThanOtherParent`]: createBooleanWrapper(false),
            });

            expect(givenFacts(factGraph).atPath(screen.agiTiebreaker, dependentId, task)).toRouteNextTo(
              screen.parentalSituation
            );
          });
          it(`when marital situation is "Divorced or separated", ask about parental support`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/nightsWithTpVsOtherParent`]: createEnumWrapper(
                `equal`,
                `/moreLessEqualOptions`
              ),
              [`/familyAndHousehold/#${dependentId}/parentalSituation`]: createEnumWrapper(
                `divorcedOrSeparated`,
                `/parentalSituationOptions`
              ),
              [`/familyAndHousehold/#${dependentId}/tpAgiHigherThanOtherParent`]: createBooleanWrapper(false),
            });

            expect(givenFacts(factGraph).atPath(screen.parentalSituation, dependentId, task)).toRouteNextTo(
              screen.parentalSupport
            );
          });
          it(`when marital situation is "Lived apart ...", ask about parental support`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/nightsWithTpVsOtherParent`]: createEnumWrapper(
                `equal`,
                `/moreLessEqualOptions`
              ),
              [`/familyAndHousehold/#${dependentId}/parentalSituation`]: createEnumWrapper(
                `livedApartLastSixMonths`,
                `/parentalSituationOptions`
              ),
              [`/familyAndHousehold/#${dependentId}/tpAgiHigherThanOtherParent`]: createBooleanWrapper(false),
            });

            expect(givenFacts(factGraph).atPath(screen.parentalSituation, dependentId, task)).toRouteNextTo(
              screen.parentalSupport
            );
          });
          it(`when marital situation is "other", exit this segment`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/nightsWithTpVsOtherParent`]: createEnumWrapper(
                `equal`,
                `/moreLessEqualOptions`
              ),
              [`/familyAndHousehold/#${dependentId}/parentalSituation`]: createEnumWrapper(
                `other`,
                `/parentalSituationOptions`
              ),
              [`/familyAndHousehold/#${dependentId}/tpAgiHigherThanOtherParent`]: createBooleanWrapper(false),
            });

            expect(givenFacts(factGraph).atPath(screen.parentalSituation, dependentId, task)).toRouteNextTo(
              exitSegment
            );
          });
          it(`when parental support is "Yes" and filer is custodial parent, show appropriate ack screen`, ({
            task,
          }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/nightsWithTpVsOtherParent`]: createEnumWrapper(
                `equal`,
                `/moreLessEqualOptions`
              ),
              [`/familyAndHousehold/#${dependentId}/parentalSituation`]: createEnumWrapper(
                `livedApartLastSixMonths`,
                `/parentalSituationOptions`
              ),
              [`/familyAndHousehold/#${dependentId}/tpAgiHigherThanOtherParent`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/parentalSupport`]: createBooleanWrapper(true),
            });

            expect(givenFacts(factGraph).atPath(screen.parentalSupport, dependentId, task)).toRouteNextTo(
              screen.custodialParentAck
            );
          });
          it(
            `when parental support is "Yes" and filer is noncustodial parent, but says they signed from 8332 ` +
              `show the approrpriate knockout`,
            ({ task }) => {
              const { factGraph } = setupFactGraphDeprecated({
                ...graphSettings,
                [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
                [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                  createBooleanWrapper(true),
                [`/familyAndHousehold/#${dependentId}/nightsWithTpVsOtherParent`]: createEnumWrapper(
                  `equal`,
                  `/moreLessEqualOptions`
                ),
                [`/familyAndHousehold/#${dependentId}/parentalSituation`]: createEnumWrapper(
                  `livedApartLastSixMonths`,
                  `/parentalSituationOptions`
                ),
                [`/familyAndHousehold/#${dependentId}/tpAgiHigherThanOtherParent`]: createBooleanWrapper(false),
                [`/familyAndHousehold/#${dependentId}/parentalSupport`]: createBooleanWrapper(true),
              });

              expect(givenFacts(factGraph).atPath(screen.parentalSupport, dependentId, task)).toRouteNextTo(
                screen.conflictingInfoKnockout
              );
            }
          );
          it(`when parental support is "No", exit this segment`, ({ task }) => {
            const { factGraph } = setupFactGraphDeprecated({
              ...graphSettings,
              [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/livedWithTpOrOtherBiologicalOrAdoptiveParentMoreThanSixMonths`]:
                createBooleanWrapper(true),
              [`/familyAndHousehold/#${dependentId}/nightsWithTpVsOtherParent`]: createEnumWrapper(
                `equal`,
                `/moreLessEqualOptions`
              ),
              [`/familyAndHousehold/#${dependentId}/parentalSituation`]: createEnumWrapper(
                `livedApartLastSixMonths`,
                `/parentalSituationOptions`
              ),
              [`/familyAndHousehold/#${dependentId}/tpAgiHigherThanOtherParent`]: createBooleanWrapper(false),
              [`/familyAndHousehold/#${dependentId}/parentalSupport`]: createBooleanWrapper(false),
            });

            expect(givenFacts(factGraph).atPath(screen.parentalSupport, dependentId, task)).toRouteNextTo(exitSegment);
          });
        });
      };
    });

    describe(`When dependent is subject to custody ...`, () => {
      const relationshipScenarios = [
        { relationshipCategory: `childOrDescendants`, relationship: `stepChild` },
        { relationshipCategory: `childOrDescendants`, relationship: `fosterChild` },
        { relationshipCategory: `childOrDescendants`, relationship: `grandChildOrOtherDescendantOfChild` },
        { relationshipCategory: `siblingOrDescendants`, relationship: `sibling` },
        { relationshipCategory: `siblingOrDescendants`, relationship: `childOfSibling` },
        { relationshipCategory: `siblingOrDescendants`, relationship: `halfSibling` },
        { relationshipCategory: `siblingOrDescendants`, relationship: `childOfHalfSibling` },
        { relationshipCategory: `siblingOrDescendants`, relationship: `stepSibling` },
        { relationshipCategory: `siblingOrDescendants`, relationship: `childOfStepSibling` },
        { relationshipCategory: `siblingOrDescendants`, relationship: `otherDescendantOfSibling` },
      ];

      const residencyScenarios = [`allYear`, `sixToElevenMonths`];

      const ageScenarios = [
        { dob: dobJustShortOf18, desc: `just short of 18`, screen: screen.notInCustodyOtherRel },
        // The following scenario skips over the benefits split segment because age 21 on July 2 fails entry condition.
        // TODO check with designers to confirm this understanding of how entry tests combine their logic.
        { dob: dob.barelyTwentyOneOnJulySecond, desc: `barely 21`, screen: screen.livedWithTpInUs },
      ];

      for (const relationship of relationshipScenarios) {
        describe(`... as the filer's ${relationship.relationship}`, () => {
          for (const residency of residencyScenarios) {
            describe(`residing with them ${residency}`, () => {
              for (const age of ageScenarios) {
                describe(`with age ${age.desc}`, () => {
                  const facts = {
                    ...baseDependentData,
                    [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
                      relationship.relationshipCategory,
                      `/relationshipCategoryOptions`
                    ),
                    ...(relationship.relationshipCategory === `childOrDescendants` && {
                      [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
                        relationship.relationship,
                        `/childRelationshipOptions`
                      ),
                    }),
                    ...(relationship.relationshipCategory === `siblingOrDescendants` && {
                      [`/familyAndHousehold/#${dependentId}/siblingRelationship`]: createEnumWrapper(
                        relationship.relationship,
                        `/siblingRelationshipOptions`
                      ),
                    }),
                    [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
                      residency,
                      `/residencyDurationOptions`
                    ),
                    [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(age.dob),
                    [`/familyAndHousehold/#${dependentId}/fullTimeStudent`]: createBooleanWrapper(true),
                    [`/familyAndHousehold/#${dependentId}/permanentTotalDisability`]: createBooleanWrapper(false),
                  };

                  it(`goes to the correct screen`, ({ task }) => {
                    const { factGraph } = setupFactGraphDeprecated(facts);
                    expect(givenFacts(factGraph).atPath(screen.residencyDuration, dependentId, task)).toRouteNextTo(
                      age.screen
                    );
                  });

                  if (age.screen === screen.areParentsLiving) {
                    for (const parentScenario of [
                      { living: true, nextScreen: screen.parentNotClaiming },
                      { living: false, nextScreen: screen.livedWithTpInUs },
                    ]) {
                      it(`then navigates correctly when parents are ${parentScenario.living ? `` : `not `}living`, ({
                        task,
                      }) => {
                        const { factGraph } = setupFactGraphDeprecated({
                          ...facts,
                          [`/familyAndHousehold/#${dependentId}/biologicalOrAdoptiveParentsLiving`]:
                            createBooleanWrapper(parentScenario.living),
                        });
                        expect(givenFacts(factGraph).atPath(screen.areParentsLiving, dependentId, task)).toRouteNextTo(
                          parentScenario.nextScreen
                        );
                      });
                    }
                  }
                });
              }
              describe(`with age 21 and disability`, () => {
                const facts = {
                  ...baseDependentData,
                  [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
                    relationship.relationshipCategory,
                    `/relationshipCategoryOptions`
                  ),
                  ...(relationship.relationshipCategory === `childOrDescendants` && {
                    [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
                      relationship.relationship,
                      `/childRelationshipOptions`
                    ),
                  }),
                  ...(relationship.relationshipCategory === `siblingOrDescendants` && {
                    [`/familyAndHousehold/#${dependentId}/siblingRelationship`]: createEnumWrapper(
                      relationship.relationship,
                      `/siblingRelationshipOptions`
                    ),
                  }),
                  [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
                    residency,
                    `/residencyDurationOptions`
                  ),
                  [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(
                    dob.barelyTwentyOneOnJulySecond
                  ),
                  [`/familyAndHousehold/#${dependentId}/permanentTotalDisability`]: createBooleanWrapper(true),
                };

                it(`goes to the correct screen`, ({ task }) => {
                  const { factGraph } = setupFactGraphDeprecated(facts);
                  expect(givenFacts(factGraph).atPath(screen.residencyDuration, dependentId, task)).toRouteNextTo(
                    screen.notInCustodyOtherRel
                  );
                });
              });
            });
          }
        });
      }
    });
  });

  describe(`The "Person's home for the year" segment`, () => {
    // The conditions for entry to this segment (livedWithTpInUs screen) have already been tested.

    const testKeepingUpHomeFlow = (maritalScenario: string, yearWidowed = ``) => {
      describe(`When dependent did live with filer in U.S. more than half of year and
       marital status is ${maritalScenario} ${yearWidowed || ``}`, () => {
        let graphSettings: { [path: string]: object } = {
          ...baseDependentData,
          [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyFourteenAtStartOfYear),
          [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
            `childOrDescendants`,
            `/relationshipCategoryOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
            `biologicalChild`,
            `/childRelationshipOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
            `allYear`,
            `/residencyDurationOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/married`]: createBooleanWrapper(false),
          [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(false),
          [`/livedApartFromSpouse`]: createBooleanWrapper(true),
          [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
          [`/familyAndHousehold/#${dependentId}/monthsLivedWithTPInUS`]: createEnumWrapper(
            `seven`,
            `/monthsLivedWithTPInUSOptions`
          ),
          [`/maritalStatus`]: createEnumWrapper(maritalScenario, `/maritalStatusOptions`),
        };

        if (yearWidowed.length) {
          graphSettings = {
            ...graphSettings,
            [`/yearOfSpouseDeath`]: createEnumWrapper(yearWidowed, `/yearOfSpouseDeathOptions`),
          };
        }
        const { factGraph } = setupFactGraphDeprecated(graphSettings);
        it(`asks about home upkeep`, ({ task }) => {
          expect(givenFacts(factGraph).atPath(screen.livedWithTpInUs, dependentId, task)).toRouteNextTo(
            screen.keepingUpHome
          );
        });
        // TODO clarify desired behavior, then test it.
        // Currently, either answer on keepingUpHome exits the segment.
        // Mural shows arrow from AddPersonKeepingUpHome to (conditionally) AddPersonKeepingUpParentsHome
      });
    };
    const maritalScenarios = enums.maritalStatus.filter((e) => e !== `married`);
    for (const maritalScenario of maritalScenarios) {
      if (maritalScenario === `widowed`) {
        enums.yearOfSpouseDeath
          .filter((e) => e !== `taxYear`)
          .forEach((e) => testKeepingUpHomeFlow(maritalScenario, e));
      } else {
        testKeepingUpHomeFlow(maritalScenario);
      }
    }
    // TODO until issue above is resolved, flow logic for AddPersonKeepingUpParentsHome is unclear.
    // Clarify that and test it.
  });

  describe(`The TIN segment`, () => {
    const exitSegment = screen.qcSupportTest;
    const tinScenarios = [
      { tinType: `ssn`, screen: screen.ssnValidForWork },
      { tinType: `itin`, screen: screen.acknowledgeTin },
      { tinType: `atin`, screen: screen.acknowledgeTin },
      { tinType: `none`, screen: exitSegment },
    ];
    for (const tinScenario of tinScenarios) {
      describe(`When dependent has ${tinScenario.tinType}`, () => {
        const { factGraph } = setupFactGraphDeprecated({
          ...baseDependentData,
          [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
            `childOrDescendants`,
            `/relationshipCategoryOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
            `biologicalChild`,
            `/childRelationshipOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyFourteenAtStartOfYear),
          [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
            `allYear`,
            `/residencyDurationOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/tinType`]: createEnumWrapper(tinScenario.tinType, `/tinTypeOptions`),
        });
        it(`shows ${tinScenario.screen}`, ({ task }) => {
          expect(givenFacts(factGraph).atPath(screen.tin, dependentId, task)).toRouteNextTo(tinScenario.screen);
        });
      });
    }

    for (const ssnScenario of enums.ssnEmploymentValidity) {
      describe(`When dependent's SSN card has ${ssnScenario}`, () => {
        const { factGraph } = setupFactGraphDeprecated({
          ...baseDependentData,
          ...baseDependentData,
          [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
            `childOrDescendants`,
            `/relationshipCategoryOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
            `biologicalChild`,
            `/childRelationshipOptions`
          ),
          [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyFourteenAtStartOfYear),
          [`/familyAndHousehold/#${dependentId}/tinType`]: createEnumWrapper(`ssn`, `/tinTypeOptions`),
          [`/familyAndHousehold/#${dependentId}/ssnEmploymentValidity`]: createEnumWrapper(
            ssnScenario,
            `/familyAndHouseholdSsnEmploymentValidityOptions`
          ),
        });
        it(`shows acknowledgement screen`, ({ task }) => {
          expect(givenFacts(factGraph).atPath(screen.ssnValidForWork, dependentId, task)).toRouteNextTo(
            screen.acknowledgeTin
          );
        });
      });
    }
  });
  // describe(`The "Support tests" segment`, ({task}) => {
  // TODO see flow.xml comment about issues to resolve in this segment.
  // Need to write tests after that.
  // });
  describe(`The "Marital status + joint return test" segment`, () => {
    const exitSegment = screen.citizenship;
    const baseMarriedData = {
      ...baseDependentData,
      [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
        `childOrDescendants`,
        `/relationshipCategoryOptions`
      ),
      [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
        `biologicalChild`,
        `/childRelationshipOptions`
      ),
      [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyTwentyOneOnJulySecond),
    };
    describe(`When dependent is married`, () => {
      const { factGraph } = setupFactGraphDeprecated({
        ...baseMarriedData,
        [`/familyAndHousehold/#${dependentId}/married`]: createBooleanWrapper(true),
      });
      it(`shows joint return screen`, ({ task }) => {
        expect(givenFacts(factGraph).atPath(screen.married, dependentId, task)).toRouteNextTo(screen.jointReturn);
      });
    });
    describe(`When dependent is not married`, () => {
      const { factGraph } = setupFactGraphDeprecated({
        ...baseMarriedData,
        [`/familyAndHousehold/#${dependentId}/married`]: createBooleanWrapper(false),
      });
      it(`exits this segment`, ({ task }) => {
        expect(givenFacts(factGraph).atPath(screen.married, dependentId, task)).toRouteNextTo(exitSegment);
      });
    });
    describe(`When married dependent is wants a joint return`, () => {
      const { factGraph } = setupFactGraphDeprecated({
        ...baseMarriedData,
        [`/familyAndHousehold/#${dependentId}/married`]: createBooleanWrapper(true),
        [`/familyAndHousehold/#${dependentId}/writableJointReturn`]: createBooleanWrapper(true),
      });
      it(`asks if they're required to file`, ({ task }) => {
        expect(givenFacts(factGraph).atPath(screen.jointReturn, dependentId, task)).toRouteNextTo(
          screen.requiredToFile
        );
      });
    });
    describe(`When married dependent is wants a joint return and is not required to file`, () => {
      const { factGraph } = setupFactGraphDeprecated({
        ...baseMarriedData,
        [`/familyAndHousehold/#${dependentId}/married`]: createBooleanWrapper(true),
        [`/familyAndHousehold/#${dependentId}/writableRequiredToFile`]: createBooleanWrapper(false),
        [`/familyAndHousehold/#${dependentId}/writableFilingOnlyForRefund`]: createBooleanWrapper(true),
        [`/familyAndHousehold/#${dependentId}/writableJointReturn`]: createBooleanWrapper(true),
      });
      it(`asks if they're filing only for a refund`, ({ task }) => {
        expect(givenFacts(factGraph).atPath(screen.requiredToFile, dependentId, task)).toRouteNextTo(
          screen.filingForRefundOnly
        );
      });
    });
    describe(`When married dependent is not filing joint return`, () => {
      const { factGraph } = setupFactGraphDeprecated({
        ...baseMarriedData,
        [`/familyAndHousehold/#${dependentId}/married`]: createBooleanWrapper(true),
        [`/familyAndHousehold/#${dependentId}/writableJointReturn`]: createBooleanWrapper(false),
      });
      it(`exits this segment`, ({ task }) => {
        expect(givenFacts(factGraph).atPath(screen.jointReturn, dependentId, task)).toRouteNextTo(exitSegment);
      });
    });

    for (const forRefundOnly of [true, false]) {
      describe(`When married dependent is ${forRefundOnly ? `` : `not `}filing for refund only`, () => {
        const { factGraph } = setupFactGraphDeprecated({
          ...baseMarriedData,
          [`/familyAndHousehold/#${dependentId}/married`]: createBooleanWrapper(true),
          [`/familyAndHousehold/#${dependentId}/writableJointReturn`]: createBooleanWrapper(true),
          [`/familyAndHousehold/#${dependentId}/writableFilingOnlyForRefund`]: createBooleanWrapper(forRefundOnly),
        });
        it(`exits this segment`, ({ task }) => {
          expect(givenFacts(factGraph).atPath(screen.filingForRefundOnly, dependentId, task)).toRouteNextTo(
            exitSegment
          );
        });
      });
    }
  });

  // N.B. (Amark): I'm skeptical that these tests are set up correctly -- in almost all cases, the person provides their
  // own support, so they could never be anyone's QC. Our exit segment is "you don't qualify for anything".
  // I think they're useful within the section, but think they'd need fair work to be useful between sections

  // This section answers serves two purposes:
  // 1. When someone is the QC for a TP, it tells us whether there's a
  //    tiebreak situation and sends the TP there (first question only)
  // 2. If someone is not yet determined to be QC or QR, if they answer
  //    "yes" to the first question, we will ask subsequent questions to
  //    assess QR status
  describe(`The "Qualifying child of another" segment`, () => {
    const exitSegment = screen.notQualifiedForAny;

    const qcOfAnotherBaseDependent = {
      ...baseDependentData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
      [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyFourteenAtStartOfYear),
      [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
        `childOrDescendants`,
        `/relationshipCategoryOptions`
      ),
      [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
        `biologicalChild`,
        `/childRelationshipOptions`
      ),
      [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(true),
      [`/familyAndHousehold/#${dependentId}/married`]: createBooleanWrapper(false),
      [`/familyAndHousehold/#${dependentId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(false),
      [`/familyAndHousehold/#${dependentId}/grossIncomeTest`]: createBooleanWrapper(false),
      [`/familyAndHousehold/#${dependentId}/tinType`]: createEnumWrapper(`ssn`, `/tinTypeOptions`),
      [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
        `allYear`,
        `/residencyDurationOptions`
      ),
      [`/familyAndHousehold/#${dependentId}/hasOtherBiologicalOrAdoptiveParent`]: createBooleanWrapper(false),
      [`/familyAndHousehold/#${baseDependentId}/monthsLivedWithTPInUS`]: createEnumWrapper(
        `twelve`,
        `/monthsLivedWithTPInUSOptions`
      ),
      [`/familyAndHousehold/#${baseDependentId}/tpPaidMostOfHomeUpkeep`]: createBooleanWrapper(true),
      [`/familyAndHousehold/#${baseDependentId}/ssnEmploymentValidity`]: createEnumWrapper(
        `neither`,
        `/familyAndHouseholdSsnEmploymentValidityOptions`
      ),
    };

    const onQRpath = {
      ...qcOfAnotherBaseDependent,
      [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.barelyFourteenAtStartOfYear),
      [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
        `lessThanSixMonths`,
        `/residencyDurationOptions`
      ),
      // eslint-disable-next-line max-len
      [`/familyAndHousehold/#${dependentId}/inParentsCustody`]: createBooleanWrapper(false), // disqualifies them from SBS
      [`/familyAndHousehold/#${dependentId}/nightsWithTpVsOtherParent`]: createEnumWrapper(
        `more`,
        `/moreLessEqualOptions`
      ),
      // Person is an elgible QR
      [`/familyAndHousehold/#${dependentId}/writableQrSupportTest`]: createBooleanWrapper(true),
      [`/familyAndHousehold/#${dependentId}/grossIncomeTest`]: createBooleanWrapper(true),
      [`/livedApartFromSpouse`]: createBooleanWrapper(true),
      [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
    };

    describe(`if the person is a potential QR`, () => {
      it(`and no one else qualifies to claim dependent as QC, it exits this segment`, ({ task }) => {
        const { factGraph } = setupFactGraphDeprecated({
          ...onQRpath,
          [`/familyAndHousehold/#${dependentId}/writableCouldBeQualifyingChildOfAnother`]: createBooleanWrapper(false),
        });
        expect(givenFacts(factGraph).atPath(screen.qcOfAnother, dependentId, task)).toRouteNextTo(
          screen.qualifiedDependent
        );
      });
      it(`and another person qualifies to claim dependent as QC, it moves you to the filing requirement screen`, ({
        task,
      }) => {
        const { factGraph } = setupFactGraphDeprecated({
          ...onQRpath,
          [`/familyAndHousehold/#${dependentId}/writableCouldBeQualifyingChildOfAnother`]: createBooleanWrapper(true),
        });
        expect(givenFacts(factGraph).atPath(screen.qcOfAnother, dependentId, task)).toRouteNextTo(
          screen.qcOfAnotherClaimerFilingRequirement
        );
      });
      it(`when other qualified person is required to file, it exits this segment`, ({ task }) => {
        const { factGraph } = setupFactGraphDeprecated({
          ...onQRpath,
          [`/familyAndHousehold/#${dependentId}/writableCouldBeQualifyingChildOfAnother`]: createBooleanWrapper(true),
          [`/familyAndHousehold/#${dependentId}/writablePotentialClaimerMustFile`]: createBooleanWrapper(true),
        });
        // TODO exiting segment is a simplification of expected behavior
        // revist this when Determination is built out
        expect(
          givenFacts(factGraph).atPath(screen.qcOfAnotherClaimerFilingRequirement, dependentId, task)
        ).toRouteNextTo(exitSegment);
      });
      it(`when other qualified person is not required to file, it asks the next question in the flow`, ({ task }) => {
        const { factGraph } = setupFactGraphDeprecated({
          ...onQRpath,
          [`/familyAndHousehold/#${dependentId}/writableCouldBeQualifyingChildOfAnother`]: createBooleanWrapper(true),
          [`/familyAndHousehold/#${dependentId}/writablePotentialClaimerMustFile`]: createBooleanWrapper(false),
        });
        expect(
          givenFacts(factGraph).atPath(screen.qcOfAnotherClaimerFilingRequirement, dependentId, task)
        ).toRouteNextTo(screen.qcOfAnotherClaimerFiling);
      });
      it(`when other qualified person chooses to file, ask if they choose to file`, ({ task }) => {
        const { factGraph } = setupFactGraphDeprecated({
          ...onQRpath,
          [`/familyAndHousehold/#${dependentId}/writableCouldBeQualifyingChildOfAnother`]: createBooleanWrapper(true),
          [`/familyAndHousehold/#${dependentId}/writablePotentialClaimerMustFile`]: createBooleanWrapper(false),
          [`/familyAndHousehold/#${dependentId}/writablePotentialClaimerDidFile`]: createBooleanWrapper(true),
        });
        expect(givenFacts(factGraph).atPath(screen.qcOfAnotherClaimerFiling, dependentId, task)).toRouteNextTo(
          screen.qcOfAnotherClaimersReturn
        );
      });
      it(`when other qualified person chooses not to file, it shows summary: not QC of other parent`, ({ task }) => {
        const { factGraph } = setupFactGraphDeprecated({
          ...onQRpath,
          [`/familyAndHousehold/#${dependentId}/writableCouldBeQualifyingChildOfAnother`]: createBooleanWrapper(true),
          [`/familyAndHousehold/#${dependentId}/writablePotentialClaimerMustFile`]: createBooleanWrapper(false),
          [`/familyAndHousehold/#${dependentId}/writablePotentialClaimerDidFile`]: createBooleanWrapper(false),
        });
        expect(givenFacts(factGraph).atPath(screen.qcOfAnotherClaimerFiling, dependentId, task)).toRouteNextTo(
          screen.qcOfAnotherSummary
        );
      });
      it(`when other qualified person chooses to file only for refund, it shows summary: not QC of other parent`, ({
        task,
      }) => {
        const { factGraph } = setupFactGraphDeprecated({
          ...onQRpath,
          [`/familyAndHousehold/#${dependentId}/writableCouldBeQualifyingChildOfAnother`]: createBooleanWrapper(true),
          [`/familyAndHousehold/#${dependentId}/writablePotentialClaimerMustFile`]: createBooleanWrapper(false),
          [`/familyAndHousehold/#${dependentId}/writablePotentialClaimerDidFile`]: createBooleanWrapper(true),
          [`/familyAndHousehold/#${dependentId}/writablePotentialClaimerFiledOnlyForRefund`]:
            createBooleanWrapper(true),
        });
        expect(givenFacts(factGraph).atPath(screen.qcOfAnotherClaimersReturn, dependentId, task)).toRouteNextTo(
          screen.qcOfAnotherSummary
        );
      });
      it(`when other qualified person chooses to file but not only for refund, it exits this segment`, ({ task }) => {
        const { factGraph } = setupFactGraphDeprecated({
          ...onQRpath,
          [`/familyAndHousehold/#${dependentId}/writableCouldBeQualifyingChildOfAnother`]: createBooleanWrapper(true),
          [`/familyAndHousehold/#${dependentId}/writablePotentialClaimerMustFile`]: createBooleanWrapper(false),
          [`/familyAndHousehold/#${dependentId}/writablePotentialClaimerDidFile`]: createBooleanWrapper(true),
          [`/familyAndHousehold/#${dependentId}/writablePotentialClaimerFiledOnlyForRefund`]:
            createBooleanWrapper(false),
        });
        expect(givenFacts(factGraph).atPath(screen.qcOfAnotherClaimersReturn, dependentId, task)).toRouteNextTo(
          exitSegment
        );
      });
    });
    describe(`if the person qualifies as a QC`, () => {
      for (const qcOfAnotherAnswer of [
        {
          hasQCOfAnother: true,
          expectedScreen: screen.qualifiedQCOfMultipleTps,
          expectedOutput: `qualified QC of multiple taxpayers screen`,
        },
        {
          hasQCOfAnother: false,
          expectedScreen: screen.qualifiedDependent,
          expectedOutput: `qualified dependent screen`,
        },
      ]) {
        it(`should ask add-person-qc-of-another and when the answer is ${qcOfAnotherAnswer.hasQCOfAnother}, \
          it should take the TP to ${qcOfAnotherAnswer.expectedOutput}`, ({ task }) => {
          const { factGraph } = setupFactGraphDeprecated({
            ...qcOfAnotherBaseDependent,
            [`/familyAndHousehold/#${dependentId}/writableCouldBeQualifyingChildOfAnother`]: createBooleanWrapper(
              qcOfAnotherAnswer.hasQCOfAnother
            ),
          });

          expect(givenFacts(factGraph).atPath(screen.qcOfAnother, dependentId, task)).toRouteNextTo(
            qcOfAnotherAnswer.expectedScreen
          );
        });
      }
    });
    describe(`when no other parent qualifies to claim qualified, but not-yet-claimed dependent as QC`, () => {
      it(`goes to qualified dependent page`, ({ task }) => {
        const { factGraph } = setupFactGraphDeprecated({
          ...qcOfAnotherBaseDependent,
          [`/familyAndHousehold/#${dependentId}/writableCouldBeQualifyingChildOfAnother`]: createBooleanWrapper(false),
        });

        expect(givenFacts(factGraph).atPath(screen.qcOfAnother, dependentId, task)).toRouteNextTo(
          screen.qualifiedDependent
        );
      });
    });
  });

  describe(`The result screens`, () => {
    const claimableDependentGraph = {
      ...baseDependentData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
      [`/familyAndHousehold/#${dependentId}/relationshipCategory`]: createEnumWrapper(
        `childOrDescendants`,
        `/relationshipCategoryOptions`
      ),
      [`/familyAndHousehold/#${dependentId}/childRelationship`]: createEnumWrapper(
        `biologicalChild`,
        `/childRelationshipOptions`
      ),
      [`/familyAndHousehold/#${dependentId}/hasOtherBiologicalOrAdoptiveParent`]: createBooleanWrapper(false),
      [`/familyAndHousehold/#${dependentId}/deceased`]: createBooleanWrapper(false),
      [`/familyAndHousehold/#${dependentId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/familyAndHousehold/#${dependentId}/residencyDuration`]: createEnumWrapper(
        `allYear`,
        `/residencyDurationOptions`
      ),
      [`/familyAndHousehold/#${dependentId}/dateOfBirth`]: createDayWrapper(dob.justShortOfFourteenAtStartOfYear),
      [`/familyAndHousehold/#${dependentId}/tpPaidMostOfHomeUpkeep`]: createBooleanWrapper(true),
      [`/familyAndHousehold/#${dependentId}/writableCouldBeQualifyingChildOfAnother`]: createBooleanWrapper(false),
      [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(false),
      [`/familyAndHousehold/#${dependentId}/married`]: createBooleanWrapper(false),
      [`/familyAndHousehold/#${dependentId}/tinType`]: createEnumWrapper(`ssn`, `/tinTypeOptions`),
      [`/familyAndHousehold/#${dependentId}/ssnEmploymentValidity`]: createEnumWrapper(
        `neither`,
        `/familyAndHouseholdSsnEmploymentValidityOptions`
      ),
      [`/familyAndHousehold/#${dependentId}/monthsLivedWithTPInUS`]: createEnumWrapper(
        `twelve`,
        `/monthsLivedWithTPInUSOptions`
      ),
    };
    const unclaimableDependentGraph = {
      ...claimableDependentGraph,
      [`/familyAndHousehold/#${dependentId}/ownSupport`]: createBooleanWrapper(true),
    };
    it(`shows qualified-dependent if the dependent is claimable`, ({ task }) => {
      const { factGraph } = setupFactGraphDeprecated(claimableDependentGraph);
      expect(givenFacts(factGraph).atPath(screen.qcOfAnother, dependentId, task)).toRouteNextTo(
        screen.qualifiedDependent
      );
    });
    it(`show potential-qp if the qp might qualify tp for benefits but we don't have enough information to know`, ({
      task,
    }) => {
      const { factGraph } = setupFactGraphDeprecated(unclaimableDependentGraph);
      expect(givenFacts(factGraph).atPath(screen.qcOfAnother, dependentId, task)).toRouteNextTo(screen.potentialQP);
    });
    it(`show potential-qp even if there is one confirmed QP`, ({ task }) => {
      const claimableDependentId = `37ce8a31-4248-42f8-9999-5754c51ab29b`;

      const twoDependentGraph = {
        ...unclaimableDependentGraph,
        '/familyAndHousehold': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [dependentId, claimableDependentId] },
        },
        [`/familyAndHousehold/#${claimableDependentId}/firstName`]: createStringWrapper(`John`),
        [`/familyAndHousehold/#${claimableDependentId}/middleInitial`]: createStringWrapper(`2`),
        [`/familyAndHousehold/#${claimableDependentId}/lastName`]: createStringWrapper(`Dependent`),
        [`/familyAndHousehold/#${claimableDependentId}/relationshipCategory`]: createEnumWrapper(
          `childOrDescendants`,
          `/relationshipCategoryOptions`
        ),
        [`/familyAndHousehold/#${claimableDependentId}/childRelationship`]: createEnumWrapper(
          `biologicalChild`,
          `/childRelationshipOptions`
        ),
        [`/familyAndHousehold/#${claimableDependentId}/hasOtherBiologicalOrAdoptiveParent`]:
          createBooleanWrapper(false),
        [`/familyAndHousehold/#${claimableDependentId}/deceased`]: createBooleanWrapper(false),
        [`/familyAndHousehold/#${claimableDependentId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
        [`/familyAndHousehold/#${claimableDependentId}/residencyDuration`]: createEnumWrapper(
          `allYear`,
          `/residencyDurationOptions`
        ),
        [`/familyAndHousehold/#${claimableDependentId}/dateOfBirth`]: createDayWrapper(
          dob.justShortOfFourteenAtStartOfYear
        ),
        [`/familyAndHousehold/#${claimableDependentId}/tpPaidMostOfHomeUpkeep`]: createBooleanWrapper(true),
        [`/familyAndHousehold/#${claimableDependentId}/writableCouldBeQualifyingChildOfAnother`]:
          createBooleanWrapper(false),
        [`/familyAndHousehold/#${claimableDependentId}/ownSupport`]: createBooleanWrapper(false),
        [`/familyAndHousehold/#${claimableDependentId}/married`]: createBooleanWrapper(false),
        [`/familyAndHousehold/#${claimableDependentId}/tinType`]: createEnumWrapper(`ssn`, `/tinTypeOptions`),
        [`/familyAndHousehold/#${claimableDependentId}/ssnEmploymentValidity`]: createEnumWrapper(
          `neither`,
          `/familyAndHouseholdSsnEmploymentValidityOptions`
        ),
        [`/familyAndHousehold/#${claimableDependentId}/monthsLivedWithTPInUS`]: createEnumWrapper(
          `twelve`,
          `/monthsLivedWithTPInUSOptions`
        ),
      };
      const { factGraph } = setupFactGraphDeprecated(twoDependentGraph);
      // confirm we have a claimable one
      expect(givenFacts(factGraph).atPath(screen.qcOfAnother, claimableDependentId, task)).toRouteNextTo(
        screen.qualifiedDependent
      );
      // and that that doesn't affect the other one
      expect(givenFacts(factGraph).atPath(screen.qcOfAnother, dependentId, task)).toRouteNextTo(screen.potentialQP);
    });
    it(`shows confirmed-qp if the qp qualifies the definitely qualifies the TP for a benefit`, ({ task }) => {
      // This creates a filer who is eligible for EITC with the same dependent as above
      const { factGraph } = setupFactGraphDeprecated({
        ...unclaimableDependentGraph,
        ...makeW2Data(THRESHOLDS.INELIGIBLE_INCOME_EITC_0QC - 1),
        '/interestReports': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [] },
        },
        '/socialSecurityReports': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [] },
        },
        '/form1099Gs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [] },
        },
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [] },
        },
        [`/maritalStatus`]: createEnumWrapper(`single`, `/maritalStatusOptions`),
        [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`1995-01-01`),
        [`/filingStatus`]: createEnumWrapper(`single`, `/filingStatusOptions`),
        [`/receivedImproperClaims`]: createBooleanWrapper(false),
        [`/eitcHadImproperClaims`]: createBooleanWrapper(false),
        [`/wasK12Educators`]: createEnumWrapper(`neither`, `/k12EducatorOptions`),
        [`/hadStudentLoanInterestPayments`]: createBooleanWrapper(false),
        [`/eitcQcOfAnother`]: createBooleanWrapper(false),
        [`/hasForeignAccounts`]: createBooleanWrapper(false),
        [`/isForeignTrustsGrantor`]: createBooleanWrapper(false),
        [`/hasForeignTrustsTransactions`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(screen.qcOfAnother, dependentId, task)).toRouteNextTo(screen.confirmedQP);
    });
  });
});
