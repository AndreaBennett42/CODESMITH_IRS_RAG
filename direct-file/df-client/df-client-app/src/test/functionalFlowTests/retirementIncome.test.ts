import { describe, expect, it } from 'vitest';
import {
  createBooleanWrapper,
  createDayWrapper,
  createDollarWrapper,
  createEnumWrapper,
  createStringWrapper,
} from '../persistenceWrappers.js';
import { setupFactGraph } from '../setupFactGraph.js';
import { baseFilerData, primaryFilerId, spouseId } from '../testData.js';
import en from '../../locales/en.yaml';
import { createFlowConfig } from '../../flow/flowConfig.js';
import makeGivenFacts from './functionalFlowUtils.js';
import flowNodes from '../../flow/flow.js';

const flow = createFlowConfig(flowNodes);
const givenFacts = makeGivenFacts(flow);

describe(`The retirement 1099-R loop`, () => {
  const formId = `0b1e355e-3d19-415d-8470-fbafd9f58361`;
  const formId2 = `ab1e355e-3d19-415d-8470-fbafd9f58362`;
  const baseIncomeData = {
    ...baseFilerData,
    '/form1099Rs': { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [formId] } },
  };

  const path = `/flow/income/retirement`;
  describe(`When filing status is ...`, () => {
    for (const key of Object.keys(en.fields[`/filingStatusOptions`])) {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/filer`]: {
          item: {
            id: `${primaryFilerId}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/filingStatus`]: createEnumWrapper(key, `/filingStatusOptions`),
      });
      const isMfj = key === `marriedFilingJointly`;
      it(`${key}, it ${isMfj ? `asks` : `does not ask`} whose 1099-R is to be entered`, ({ task }) => {
        expect(givenFacts(factGraph).atPath(`${path}/1099-r-nonstandard`, formId, task)).toRouteNextTo(
          isMfj ? `${path}/1099-r-add-whose-1099-r` : `${path}/1099-r-add-recipient-name`
        );
      });
    }
  });
  describe(`When 1099-R address matches tax return`, () => {
    it(`navigates to payer info screen`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/recipientAddressChoice`]: createEnumWrapper(
          `matchesReturn`,
          `/recipientAddressChoiceOptions`
        ),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-recipient-address-choice`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-payer-info`
      );
    });
  });
  describe(`When 1099-R address does not match tax return`, () => {
    it(`navigates to address entry screen, then payer info screen`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/recipientAddressChoice`]: createEnumWrapper(
          `different`,
          `/recipientAddressChoiceOptions`
        ),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-recipient-address-choice`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-recipient-address`
      );
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-recipient-address`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-payer-info`
      );
    });
    describe(`When the user enters information in box 6`, () => {
      it(`navigates to knockout screen when amount is more than zero`, ({ task }) => {
        const { factGraph } = setupFactGraph({
          ...baseIncomeData,
          [`/form1099Rs/#${formId}/writableNetAppreciation`]: createDollarWrapper(`1.00`),
        });
        expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-6`, formId, task)).toRouteNextTo(
          `${path}/1099-r-box-6-ko`
        );
      });
    });
  });
  describe(`When the user enters information in box 2b`, () => {
    it(`navigates to box 3 when Taxable amount not determined checkbox is unchecked `, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableTaxableAmount`]: createDollarWrapper(`5.00`),
        [`/form1099Rs/#${formId}/writableTaxableAmountNotDetermined`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-2b`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-3`
      );
    });

    it(`navigates to knockout screen when the Taxable amount not determined checkbox is checked`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableTaxableAmountNotDetermined`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-2b`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-2b-ko-1`
      );
    });

    it(`navigates to box 3 when the Taxable amount is not blank
        and Taxable amount not determined checkbox is unchecked`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableTaxableAmount`]: createDollarWrapper(`5.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-2b`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-3`
      );
    });

    it(`navigates to box 3 when Total distribution is unchecked and the TP was born after 1/2/1936`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`1957-01-01`),
        [`/form1099Rs/#${formId}/writableTotalDistribution`]: createBooleanWrapper(false),
        [`/form1099Rs/#${formId}/writableTaxableAmount`]: createDollarWrapper(`5.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-2b`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-3`
      );
    });

    it(`navigates to box 3 when Total distribution is unchecked and the TP was born before 1/2/1936`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`1935-01-01`),
        [`/form1099Rs/#${formId}/writableTotalDistribution`]: createBooleanWrapper(false),
        [`/form1099Rs/#${formId}/writableTaxableAmount`]: createDollarWrapper(`5.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-2b`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-3`
      );
    });

    it(`navigates to box 3 when Total distribution is checked and the TP was born after 1/2/1936`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`1957-01-01`),
        [`/form1099Rs/#${formId}/writableTotalDistribution`]: createBooleanWrapper(true),
        [`/form1099Rs/#${formId}/writableTaxableAmount`]: createDollarWrapper(`5.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-2b`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-3`
      );
    });

    it(`navigates to knockout when Total distribution is checked and the TP was born before 1/2/1936`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`1935-01-01`),
        [`/form1099Rs/#${formId}/writableTaxableAmountNotDetermined`]: createBooleanWrapper(false),
        [`/form1099Rs/#${formId}/writableTotalDistribution`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-2b`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-2b-ko-2`
      );
    });
  });

  describe(`When the user enters information in box 3`, () => {
    it(`navigates to knockout screen when amount is more than zero`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableCapitalGain`]: createDollarWrapper(`1.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-3`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-3-ko`
      );
    });

    it(`navigates to knockout screen when only spouse has capital gains`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableCapitalGain`]: createDollarWrapper(`0.00`),
        [`/form1099Rs/#${spouseId}/writableCapitalGain`]: createDollarWrapper(`1.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-3`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-box-3-ko`
      );
    });

    it(`navigates to box 4 screen when amount is zero`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableCapitalGain`]: createDollarWrapper(`0.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-3`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-4`
      );
    });
  });

  describe(`When the user enters information in box 6`, () => {
    it(`navigates to knockout screen when net appreciation amount is more than zero`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableNetAppreciation`]: createDollarWrapper(`1.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-6`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-6-ko`
      );
    });

    it(`navigates to knockout screen when spouse's net appreciation amount is more than zero`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableNetAppreciation`]: createDollarWrapper(`0.00`),
        [`/form1099Rs/#${spouseId}/writableNetAppreciation`]: createDollarWrapper(`1.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-6`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-box-6-ko`
      );
    });
  });
  describe(`When the user enters information in box 7`, () => {
    it(`navigates to the next screen when the distribution code is supported`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`G`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-7-codes`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-8`
      );
    });
    it(`navigates to knockout screen when the distribution code is not supported`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`0`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-7-codes`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-7-code-knockout`
      );
    });
    it(`navigates to knockout screen when any 1099R has an unsupported distribution code`, ({ task }) => {
      const supportedCode = `2`;
      const unsupportedCode = `0`;
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, formId2] },
        },

        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(supportedCode),
        [`/form1099Rs/#${formId2}/writableDistributionCode`]: createStringWrapper(unsupportedCode),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-7-codes`, formId2, task)).toRouteNextTo(
        `${path}/1099-r-box-7-code-knockout`
      );
    });
    it(`knocks Primary Filer out when the distribution code includes G and box 2a is > $0`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId] },
        },
        [`/form1099Rs/#${formId}/writableTaxableAmount`]: createDollarWrapper(`1000.00`),
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`4G`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-7-codes`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-7-pretax-to-posttax-knockout`
      );
    });
    it(`knocks Secondary Filer out when the distribution code includes G and box 2a is > $0`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableTaxableAmount`]: createDollarWrapper(`0.00`),
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`4G`),
        [`/form1099Rs/#${spouseId}/writableTaxableAmount`]: createDollarWrapper(`1000.00`),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`4G`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-7-codes`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-box-7-pretax-to-posttax-knockout`
      );
    });
    it(`knocks Primary Filer out when the distribution code includes H and box 2a is > $0`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId] },
        },
        [`/form1099Rs/#${formId}/writableTaxableAmount`]: createDollarWrapper(`1000.00`),
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`H`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-7-codes`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-7-pretax-to-posttax-knockout`
      );
    });
    it(`knocks Secondary Filer out when the distribution code includes H and box 2a is > $0`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableTaxableAmount`]: createDollarWrapper(`0.00`),
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`4H`),
        [`/form1099Rs/#${spouseId}/writableTaxableAmount`]: createDollarWrapper(`1000.00`),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`4H`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-7-codes`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-box-7-pretax-to-posttax-knockout`
      );
    });
    it(`allows Primary Filer to proceed when the distribution code includes H and box 2a is $0`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId] },
        },
        [`/form1099Rs/#${formId}/writableTaxableAmount`]: createDollarWrapper(`0.00`),
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`H`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-7-codes`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-8`
      );
    });
    it(`allows Secondary Filer to proceed when the distribution code includes H and box 2a is $0`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableTaxableAmount`]: createDollarWrapper(`0.00`),
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`4G`),
        [`/form1099Rs/#${spouseId}/writableTaxableAmount`]: createDollarWrapper(`0.00`),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`H`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-7-codes`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-8`
      );
    });
  });

  //'early distribution no known exception' code 1 is not implemented yet
  /*
  describe(`When the user has/doesn't have an early distribution`, () => {
    it(`navigates to the next screen when the Taxpayer does not have a qualified early distribution`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`1`),
        [`/form1099Rs/#${formId}/writableQualifiedEarlyDistribution`]: createBooleanWrapper(false),
      });
      expect(
        givenFacts(factGraph).atPath(`${path}/1099-r-box-7-early-distribution-question`, formId, task)
      ).toRouteNextTo(`${path}/1099-r-box-7-indirect-rollover`);
    });
    it(`navigates to the next screen when the Spouse does not have a qualified early distribution`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`1`),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`1`),
        [`/form1099Rs/#${formId}/writableQualifiedEarlyDistribution`]: createBooleanWrapper(false),
        [`/form1099Rs/#${spouseId}/writableQualifiedEarlyDistribution`]: createBooleanWrapper(false),
      });
      expect(
        givenFacts(factGraph).atPath(`${path}/1099-r-box-7-early-distribution-question`, spouseId, task)
      ).toRouteNextTo(`${path}/1099-r-box-7-indirect-rollover`);
    });
    it(`navigates to the knockout screen when the Taxpayer has a qualified arly distribution`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`1`),
        [`/form1099Rs/#${formId}/writableQualifiedEarlyDistribution`]: createBooleanWrapper(true),
      });
      expect(
        givenFacts(factGraph).atPath(`${path}/1099-r-box-7-early-distribution-question`, formId, task)
      ).toRouteNextTo(`${path}/1099-r-box-7-early-distribution-knockout`);
    });
    it(`navigates to the knockout screen when the Spouse has a qualified early distribution`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`1`),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`1`),
        [`/form1099Rs/#${formId}/writableQualifiedEarlyDistribution`]: createBooleanWrapper(false),
        [`/form1099Rs/#${spouseId}/writableQualifiedEarlyDistribution`]: createBooleanWrapper(true),
      });
      expect(
        givenFacts(factGraph).atPath(`${path}/1099-r-box-7-early-distribution-question`, spouseId, task)
      ).toRouteNextTo(`${path}/1099-r-box-7-early-distribution-knockout`);
    });
  });
  */
  describe(`When the user enters indirect rollover qualifying distribution codes in box 7`, () => {
    it(`navigates to the follow-up question when the Taxpayer's code includes '2'`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`b2`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-7-codes`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-7-indirect-rollover`
      );
    });
    it(`navigates to the follow-up question when the Spouse's code includes '2'`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`g`),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`b2`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-7-codes`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-box-7-indirect-rollover`
      );
    });
    it(`knocks out when The Taxpayer rolled over their distribution into another retirement account`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`7`),
        [`/form1099Rs/#${formId}/writableIsIndirectRollover`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-box-7-indirect-rollover`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-7-indirect-rollover-knockout`
      );
    });
    it(`knocks out when the Spouse rolled over their distribution into another retirement account`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`g`),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`7`),
        [`/form1099Rs/#${spouseId}/writableIsIndirectRollover`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-box-7-indirect-rollover`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-box-7-indirect-rollover-knockout`
      );
    });
    //eslint-disable-next-line max-len
    it(`proceeds to ask Taxpayers's distribution is is from a military retirement plan`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`7`),
        [`/form1099Rs/#${formId}/writableIsIndirectRollover`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-box-7-indirect-rollover`, formId, task)).toRouteNextTo(
        `${path}/1099-r-military-plan`
      );
    });
    // eslint-disable-next-line max-len
    it(`proceeds to ask if Spouse's distribution is from a military retirement plan`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`g`),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`7`),
        [`/form1099Rs/#${spouseId}/writableIsIndirectRollover`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-box-7-indirect-rollover`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-military-plan`
      );
    });
  });

  describe(`When the user has indirect rollovers with/without a qualified disaster with distribution code 2`, () => {
    it(`navigates to next screen when the Taxpayer has no qualified disaster dist & code is 2 and no B`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${formId}/writableQualifiedDisasterDistribution`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-disaster`, formId, task)).toRouteNextTo(`${path}/1099-r-pso`);
    });
    it(`navigates to next screen when the Taxpayer has no qualified disaster dist & code is G`, ({ task }) => {
      // This is kind of a weird case, because the TP shouldn't get to the disaster screen with a G code.
      // This is testing that the TP will be able to proceed to the next screen if they somehow get to the
      // disaster screen with a code other than 2, 3, or 7. Ex. If they enter an incorrect code and then
      // go back to correct it after completing the disaster screen.
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`G`),
        [`/form1099Rs/#${formId}/writableQualifiedDisasterDistribution`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-disaster`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-8`
      );
    });
    it(`navigates to next screen when the Spouse has no qualified disaster dist & code is 2 and no B`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${spouseId}/writableQualifiedDisasterDistribution`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-disaster`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-pso`
      );
    });
    it(`navigates to knockout screen when the Taxpayer has a qualified disaster dist & code is 2 and no B`, ({
      task,
    }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${formId}/writableQualifiedDisasterDistribution`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-disaster`, formId, task)).toRouteNextTo(
        `${path}/1099-r-disaster-knockout`
      );
    });
    it(`navigates to knockout screen when the Spouse has a qualified disaster dist & code is 2 and no B`, ({
      task,
    }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${formId}/writableQualifiedDisasterDistribution`]: createBooleanWrapper(false),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${spouseId}/writableQualifiedDisasterDistribution`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-disaster`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-disaster-knockout`
      );
    });
    it(`navigates to next screen when the Taxpayer has no qualified disaster dist & code is 2B`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2B`),
        [`/form1099Rs/#${formId}/writableQualifiedDisasterDistribution`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-disaster`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-8`
      );
    });
    it(`navigates to next screen when the Spouse has no qualified disaster dist & code is 2B`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`2B`),
        [`/form1099Rs/#${spouseId}/writableQualifiedDisasterDistribution`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-disaster`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-8`
      );
    });
    it(`navigates to knockout screen when the Taxpayer has a qualified disaster dist & code is 2B`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2B`),
        [`/form1099Rs/#${formId}/writableQualifiedDisasterDistribution`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-disaster`, formId, task)).toRouteNextTo(
        `${path}/1099-r-disaster-knockout`
      );
    });
    it(`navigates to next screen when the Spouse has a qualified disaster dist & code is 2 is B`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${formId}/writableQualifiedDisasterDistribution`]: createBooleanWrapper(false),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`2B`),
        [`/form1099Rs/#${spouseId}/writableQualifiedDisasterDistribution`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-disaster`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-disaster-knockout`
      );
    });
  });

  describe(`When the user has is/is not a public safety officer with distribution code 2 (no B)`, () => {
    it(`navigates to next screen when the Taxpayer is not a public safety officer`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${formId}/writeablePublicSafetyOfficer`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-pso`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-8`
      );
    });
    it(`navigates to next screen when the Spouse is not a public safety officer`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${formId}/writeablePublicSafetyOfficer`]: createBooleanWrapper(false),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${spouseId}/writeablePublicSafetyOfficer`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-pso`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-8`
      );
    });
    it(`navigates to next screen when the Taxpayer is a public safety officer`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${formId}/writeablePublicSafetyOfficer`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-pso`, formId, task)).toRouteNextTo(
        `${path}/1099-r-pso-election`
      );
    });
    it(`navigates to next screen when the Spouse is a public safety officer`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${formId}/writeablePublicSafetyOfficer`]: createBooleanWrapper(false),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${spouseId}/writeablePublicSafetyOfficer`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-pso`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-pso-election`
      );
    });
    it(`navigates to next screen when the Taxpayer is a public safety officer without paying premiums`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${formId}/writeablePublicSafetyOfficerPremiums`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-pso-election`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-8`
      );
    });

    it(`navigates to next screen when the Spouse is a public safety officer without paying premiums`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${formId}/writeablePublicSafetyOfficer`]: createBooleanWrapper(true),
        [`/form1099Rs/#${formId}/writeablePublicSafetyOfficerPremiums`]: createBooleanWrapper(false),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${spouseId}/writeablePublicSafetyOfficer`]: createBooleanWrapper(true),
        [`/form1099Rs/#${spouseId}/writeablePublicSafetyOfficerPremiums`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-pso-election`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-8`
      );
    });
    it(`navigates to next screen when the Taxpayer is a public safety officer with paying premiums`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${formId}/writeablePublicSafetyOfficer`]: createBooleanWrapper(true),
        [`/form1099Rs/#${formId}/writeablePublicSafetyOfficerPremiums`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-pso-election`, formId, task)).toRouteNextTo(
        `${path}/1099-r-pso-knockout`
      );
    });
    it(`navigates to next screen when the Spouse is a public safety officer with paying premiums`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${formId}/writeablePublicSafetyOfficer`]: createBooleanWrapper(true),
        [`/form1099Rs/#${formId}/writeablePublicSafetyOfficerPremiums`]: createBooleanWrapper(false),
        [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`2`),
        [`/form1099Rs/#${spouseId}/writeablePublicSafetyOfficer`]: createBooleanWrapper(true),
        [`/form1099Rs/#${spouseId}/writeablePublicSafetyOfficerPremiums`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-pso-election`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-pso-knockout`
      );
    });
  });

  describe(`When the user enters information in box 8 amount`, () => {
    it(`navigates to knockout screen when other amount in box 8 is more than zero`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableOtherDollar`]: createDollarWrapper(`1.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-8`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-8-value-knockout`
      );
    });

    it(`navigates to knockout screen when only spouse has other amount in box 8`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableOtherDollar`]: createDollarWrapper(`0.00`),
        [`/form1099Rs/#${spouseId}/writableOtherDollar`]: createDollarWrapper(`1.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-8`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-box-8-value-knockout`
      );
    });

    it(`navigates to box 9a screen when other amount in box 8 is zero`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableOtherDollar`]: createDollarWrapper(`0.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-8`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-9a`
      );
    });
  });

  describe(`When the user enters information in box 8 percentage`, () => {
    it(`navigates to knockout screen when other percentage in box 8 is more than zero`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableOtherPercentage`]: createStringWrapper(`22`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-8`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-8-percentage-knockout`
      );
    });

    it(`navigates to knockout screen when only spouse has other percentage in box 8`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableOtherPercentage`]: createStringWrapper(`0`),
        [`/form1099Rs/#${spouseId}/writableOtherPercentage`]: createStringWrapper(`22`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-8`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-box-8-percentage-knockout`
      );
    });

    it(`navigates to box 9a screen when other percentage in box 8 is zero`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableOtherPercentage`]: createStringWrapper(`0`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-8`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-9a`
      );
    });
  });

  describe(`When the user enters information in box 9a percentage`, () => {
    it(`navigates to knockout screen when other percentage in box 9a is more than zero`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writablePercentageTotalDistribution`]: createStringWrapper(`22`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-9a`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-9a-percentage-knockout`
      );
    });

    it(`navigates to knockout screen when only spouse has nonzero percentage in box 9a`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writablePercentageTotalDistribution`]: createStringWrapper(`0`),
        [`/form1099Rs/#${spouseId}/writablePercentageTotalDistribution`]: createStringWrapper(`22`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-9a`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-box-9a-percentage-knockout`
      );
    });

    it(`navigates to box 9b screen when percentage in box 9a is zero`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writablePercentageTotalDistribution`]: createStringWrapper(`0`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-9a`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-9b`
      );
    });
  });

  describe(`When the user enters information in box 10`, () => {
    it(`navigates to knockout screen when amount allocable is more than zero`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableAmountAllocableToIRR`]: createDollarWrapper(`1.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-10`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-10-knockout`
      );
    });

    it(`navigates to knockout screen when only spouse has an amount allocable within 5 years`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/writableAmountAllocableToIRR`]: createDollarWrapper(`0.00`),
        [`/form1099Rs/#${spouseId}/writableAmountAllocableToIRR`]: createDollarWrapper(`1.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-10`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-box-10-knockout`
      );
    });

    it(`navigates to box 11 screen when amount allocable is zero`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableAmountAllocableToIRR`]: createDollarWrapper(`0.00`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-10`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-11`
      );
    });
  });
  describe(`When the user enters information in box 12`, () => {
    it(`navigates to knockout screen when FACTA filing requirement box is checked`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/factaFilingRequirement`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-12`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-12-knockout`
      );
    });

    it(`navigates to knockout screen when only spouse has the FACTA filing requirement box checked`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${formId}/factaFilingRequirement`]: createBooleanWrapper(false),
        [`/form1099Rs/#${spouseId}/factaFilingRequirement`]: createBooleanWrapper(true),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-12`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-box-12-knockout`
      );
    });

    it(`navigates to box 13 screen when the FACTA filing requirement box is un checked`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/factaFilingRequirement`]: createBooleanWrapper(false),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-12`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-13`
      );
    });
  });

  describe(`On Box 13`, () => {
    it(`knocks the Taxpayer out when they enter a date of reportable death benefit`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableDateOfPayment`]: createDayWrapper(`2023-01-01`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-13`, formId, task)).toRouteNextTo(
        `${path}/1099-r-box-13-ko`
      );
    });

    it(`knocks the Spouse out when they enter a date of reportable death benefit`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
        [`/form1099Rs/#${spouseId}/writableDateOfPayment`]: createDayWrapper(`2023-01-01`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-13`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-box-13-ko`
      );
    });

    it(`Taxpayer proceeds to box 14 if box 13 is empty`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-13`, formId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-14`
      );
    });

    it(`Spouse proceeds to box 14 if box 13 is empty`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        '/form1099Rs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [formId, spouseId] },
        },
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-13`, spouseId, task)).toRouteNextTo(
        `${path}/1099-r-add-box-14`
      );
    });
  });

  describe(`When on Box 15`, () => {
    describe(`the taxpayer`, () => {
      it(`is knocked out if 1099R is from another state and Box 14 is blank and box 7 code is 7`, ({ task }) => {
        const { factGraph } = setupFactGraph({
          ...baseIncomeData,
          [`/form1099Rs/#${formId}/writablePayerState`]: createEnumWrapper(`differentState`, `/incomeFormStateOptions`),
          [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`7`),
        });
        expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-15`, formId, task)).toRouteNextTo(
          `${path}/1099-r-box-15-ko`
        );
      });
      it(`is knocked out if 1099R is from another state and Box 14 is 0.00 and box 7 code is 7`, ({ task }) => {
        const { factGraph } = setupFactGraph({
          ...baseIncomeData,
          [`/form1099Rs/#${formId}/writablePayerState`]: createEnumWrapper(`differentState`, `/incomeFormStateOptions`),
          [`/form1099Rs/#${formId}/writableStateTaxWithheld`]: createDollarWrapper(`0.00`),
          [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`7`),
        });
        expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-15`, formId, task)).toRouteNextTo(
          `${path}/1099-r-box-15-ko`
        );
      });
      it(`is knocked out if 1099R is from another state and Box 14 is 100.00 and box 7 code is G`, ({ task }) => {
        const { factGraph } = setupFactGraph({
          ...baseIncomeData,
          [`/form1099Rs/#${formId}/writablePayerState`]: createEnumWrapper(`differentState`, `/incomeFormStateOptions`),
          [`/form1099Rs/#${formId}/writableStateTaxWithheld`]: createDollarWrapper(`100.00`),
          [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`G`),
        });
        expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-15`, formId, task)).toRouteNextTo(
          `${path}/1099-r-box-15-ko`
        );
      });
      it(`can proceed if 1099R is from another state and Box 14 is 0.00 and box 7 code is G`, ({ task }) => {
        const { factGraph } = setupFactGraph({
          ...baseIncomeData,
          [`/form1099Rs/#${formId}/writablePayerState`]: createEnumWrapper(`differentState`, `/incomeFormStateOptions`),
          [`/form1099Rs/#${formId}/writableStateTaxWithheld`]: createDollarWrapper(`0.00`),
          [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`G`),
        });
        expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-15`, formId, task)).toRouteNextTo(
          `${path}/1099-r-add-boxes-16-19`
        );
      });
    });
    describe(`the spouse`, () => {
      it(`is knocked out if 1099R is from another state and Box 14 is blank and box 7 code is 7`, ({ task }) => {
        const { factGraph } = setupFactGraph({
          ...baseIncomeData,
          '/form1099Rs': {
            $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
            item: { items: [formId, spouseId] },
          },
          [`/form1099Rs/#${formId}/writablePayerState`]: createEnumWrapper(`differentState`, `/incomeFormStateOptions`),
          [`/form1099Rs/#${formId}/writableStateTaxWithheld`]: createDollarWrapper(`0.00`),
          [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`G`),
          [`/form1099Rs/#${spouseId}/writablePayerState`]: createEnumWrapper(
            `differentState`,
            `/incomeFormStateOptions`
          ),
          [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`7`),
        });
        expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-15`, spouseId, task)).toRouteNextTo(
          `${path}/1099-r-box-15-ko`
        );
      });
      it(`is knocked out if 1099R is from another state and Box 14 is 0.00 and box 7 code is 7`, ({ task }) => {
        const { factGraph } = setupFactGraph({
          ...baseIncomeData,
          '/form1099Rs': {
            $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
            item: { items: [formId, spouseId] },
          },
          [`/form1099Rs/#${formId}/writablePayerState`]: createEnumWrapper(`differentState`, `/incomeFormStateOptions`),
          [`/form1099Rs/#${formId}/writableStateTaxWithheld`]: createDollarWrapper(`0.00`),
          [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`G`),
          [`/form1099Rs/#${spouseId}/writablePayerState`]: createEnumWrapper(
            `differentState`,
            `/incomeFormStateOptions`
          ),
          [`/form1099Rs/#${spouseId}/writableStateTaxWithheld`]: createDollarWrapper(`0.00`),
          [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`7`),
        });
        expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-15`, spouseId, task)).toRouteNextTo(
          `${path}/1099-r-box-15-ko`
        );
      });
      it(`is knocked out if 1099R is from another state and Box 14 is 100.00 and box 7 code is G`, ({ task }) => {
        const { factGraph } = setupFactGraph({
          ...baseIncomeData,
          '/form1099Rs': {
            $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
            item: { items: [formId, spouseId] },
          },
          [`/form1099Rs/#${formId}/writablePayerState`]: createEnumWrapper(`differentState`, `/incomeFormStateOptions`),
          [`/form1099Rs/#${formId}/writableStateTaxWithheld`]: createDollarWrapper(`0.00`),
          [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`G`),
          [`/form1099Rs/#${spouseId}/writablePayerState`]: createEnumWrapper(
            `differentState`,
            `/incomeFormStateOptions`
          ),
          [`/form1099Rs/#${spouseId}/writableStateTaxWithheld`]: createDollarWrapper(`100.00`),
          [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`G`),
        });
        expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-15`, spouseId, task)).toRouteNextTo(
          `${path}/1099-r-box-15-ko`
        );
      });
      it(`can proceed if 1099R is from another state and Box 14 is 0.00 and box 7 code is G`, ({ task }) => {
        const { factGraph } = setupFactGraph({
          ...baseIncomeData,
          '/form1099Rs': {
            $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
            item: { items: [formId, spouseId] },
          },
          [`/form1099Rs/#${formId}/writablePayerState`]: createEnumWrapper(`differentState`, `/incomeFormStateOptions`),
          [`/form1099Rs/#${formId}/writableStateTaxWithheld`]: createDollarWrapper(`0.00`),
          [`/form1099Rs/#${formId}/writableDistributionCode`]: createStringWrapper(`G`),
          [`/form1099Rs/#${spouseId}/writablePayerState`]: createEnumWrapper(
            `differentState`,
            `/incomeFormStateOptions`
          ),
          [`/form1099Rs/#${spouseId}/writableStateTaxWithheld`]: createDollarWrapper(`0`),
          [`/form1099Rs/#${spouseId}/writableDistributionCode`]: createStringWrapper(`G`),
        });
        expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-box-15`, spouseId, task)).toRouteNextTo(
          `${path}/1099-r-add-boxes-16-19`
        );
      });
    });
  });

  describe(`When the user skips/enters information in the account number box`, () => {
    it(`navigates to the dataview screen when an account number is entered`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...baseIncomeData,
        [`/form1099Rs/#${formId}/writableAccountNumber`]: createStringWrapper(`ACCOUNT1234`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-r-add-account-number`, formId, task)).toRouteNextTo(
        `/data-view/loop/%2Fform1099Rs/${formId}`
      );
    });
  });
});
