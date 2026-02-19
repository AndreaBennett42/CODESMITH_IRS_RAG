import { it, describe, expect } from 'vitest';
import { baseFilerData, primaryFilerId } from '../testData.js';
import { createBooleanWrapper } from '../persistenceWrappers.js';
import { createFlowConfig } from '../../flow/flowConfig.js';
import flowNodes from '../../flow/flow.js';
import makeGivenFacts from './functionalFlowUtils.js';
import { setupFactGraph } from '../setupFactGraph.js';

const flow = createFlowConfig(flowNodes);
const givenFacts = makeGivenFacts(flow);

describe(`Premium Tax Credit`, () => {
  describe(`Non MFJ dependent tax payer flows and outcome`, () => {
    const updatedBaseFilersToEnableEnteringFlow = {
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
    };

    const screens = {
      notPartOfDiffTaxFamily: `/flow/credits-and-deductions/credits/dep-tp-enrollee-not-part-of-tax-family`,
      depTpKo: `/flow/credits-and-deductions/credits/dep-tp-ko`,
      depTp8962NotNeeded: `/flow/credits-and-deductions/credits/dep-tp-8962-not-needed`,
      depTpKeepGoing: `/flow/credits-and-deductions/credits/dep-tp-ptc-keep-going`,
    };

    it(`leads you to the KO screen`, ({ task }) => {
      const leadsToKo = [
        {
          isClaimed: true,
          enrolledSelf: true,
          enrolledOthers: true,
          notPartOfDifferentTaxFam: false,
        },
        {
          isClaimed: false,
          enrolledSelf: true,
          enrolledOthers: true,
          notPartOfDifferentTaxFam: true,
        },
      ];

      leadsToKo.forEach((scenario, i) => {
        const { factGraph } = setupFactGraph({
          ...updatedBaseFilersToEnableEnteringFlow,
          [`/filers/#${primaryFilerId}/willBeClaimed`]: createBooleanWrapper(scenario.isClaimed),
          '/dependentTpSelfEnrolled': createBooleanWrapper(scenario.enrolledSelf),
          '/dependentTpEnrolledSomeoneElse': createBooleanWrapper(scenario.enrolledOthers),
          '/writableDependentTpOtherMembersArePartOfDifferentTaxFamily': createBooleanWrapper(
            scenario.notPartOfDifferentTaxFam
          ),
        });
        expect(givenFacts(factGraph).atPath(screens.notPartOfDiffTaxFamily, null, task)).toRouteNextTo(
          screens.depTpKo,
          `Error in test scenario at index ${i}`
        );
      });
    });

    it(`leads to not needed screen`, ({ task }) => {
      const leadsToNotNeededScreen = [
        {
          isClaimed: true,
          enrolledSelf: false,
          enrolledOthers: false,
          notPartOfDifferentTaxFam: true,
        },
        {
          isClaimed: true,
          enrolledSelf: false,
          enrolledOthers: false,
          notPartOfDifferentTaxFam: false,
        },
        {
          isClaimed: false,
          enrolledSelf: false,
          enrolledOthers: false,
          notPartOfDifferentTaxFam: true,
        },
        {
          isClaimed: false,
          enrolledSelf: false,
          enrolledOthers: false,
          notPartOfDifferentTaxFam: false,
        },
        {
          isClaimed: true,
          enrolledSelf: true,
          enrolledOthers: false,
          notPartOfDifferentTaxFam: true,
        },
        {
          isClaimed: true,
          enrolledSelf: true,
          enrolledOthers: false,
          notPartOfDifferentTaxFam: false,
        },
        {
          isClaimed: true,
          enrolledSelf: true,
          enrolledOthers: true,
          notPartOfDifferentTaxFam: true,
        },
        {
          isClaimed: true,
          enrolledSelf: false,
          enrolledOthers: true,
          notPartOfDifferentTaxFam: true,
        },
        {
          isClaimed: false,
          enrolledSelf: false,
          enrolledOthers: true,
          notPartOfDifferentTaxFam: true,
        },
      ];

      leadsToNotNeededScreen.forEach((scenario, i) => {
        const { factGraph } = setupFactGraph({
          ...updatedBaseFilersToEnableEnteringFlow,
          [`/filers/#${primaryFilerId}/willBeClaimed`]: createBooleanWrapper(scenario.isClaimed),
          '/dependentTpSelfEnrolled': createBooleanWrapper(scenario.enrolledSelf),
          '/dependentTpEnrolledSomeoneElse': createBooleanWrapper(scenario.enrolledOthers),
          '/writableDependentTpOtherMembersArePartOfDifferentTaxFamily': createBooleanWrapper(
            scenario.notPartOfDifferentTaxFam
          ),
        });
        expect(givenFacts(factGraph).atPath(screens.notPartOfDiffTaxFamily, null, task)).toRouteNextTo(
          screens.depTp8962NotNeeded,
          `Error in test scenario at index ${i}`
        );
      });
    });

    it(`leads to keep going screen`, ({ task }) => {
      const leadsToKeepGoingScreen = [
        {
          isClaimed: false,
          enrolledSelf: true,
          enrolledOthers: false,
          notPartOfDifferentTaxFam: true,
        },
        {
          isClaimed: false,
          enrolledSelf: true,
          enrolledOthers: false,
          notPartOfDifferentTaxFam: false,
        },
        {
          isClaimed: true,
          enrolledSelf: false,
          enrolledOthers: true,
          notPartOfDifferentTaxFam: false,
        },
        {
          isClaimed: false,
          enrolledSelf: false,
          enrolledOthers: true,
          notPartOfDifferentTaxFam: false,
        },
        {
          isClaimed: false,
          enrolledSelf: true,
          enrolledOthers: true,
          notPartOfDifferentTaxFam: false,
        },
      ];

      leadsToKeepGoingScreen.forEach((scenario, i) => {
        const { factGraph } = setupFactGraph({
          ...updatedBaseFilersToEnableEnteringFlow,
          [`/filers/#${primaryFilerId}/willBeClaimed`]: createBooleanWrapper(scenario.isClaimed),
          '/dependentTpSelfEnrolled': createBooleanWrapper(scenario.enrolledSelf),
          '/dependentTpEnrolledSomeoneElse': createBooleanWrapper(scenario.enrolledOthers),
          '/writableDependentTpOtherMembersArePartOfDifferentTaxFamily': createBooleanWrapper(
            scenario.notPartOfDifferentTaxFam
          ),
        });
        expect(givenFacts(factGraph).atPath(screens.notPartOfDiffTaxFamily, null, task)).toRouteNextTo(
          screens.depTpKeepGoing,
          `Error in test scenario at index ${i}`
        );
      });
    });
  });
});
