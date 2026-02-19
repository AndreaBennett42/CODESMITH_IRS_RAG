import { it, describe, expect } from 'vitest';
import { baseFilerData } from '../testData.js';
import {
  createEnumWrapper,
  createBooleanWrapper,
  createDollarWrapper,
  createEinWrapper,
  createStringWrapper,
} from '../persistenceWrappers.js';
import en from '../../locales/en.yaml';
import { createFlowConfig } from '../../flow/flowConfig.js';
import flowNodes from '../../flow/flow.js';
import makeGivenFacts from './functionalFlowUtils.js';
import { setupFactGraph } from '../setupFactGraph.js';

const flow = createFlowConfig(flowNodes);
const givenFacts = makeGivenFacts(flow);

describe(`The \`unemployment\` subcategory`, () => {
  const path = `/flow/income/unemployment`;
  describe(`When filing status is ...`, () => {
    for (const key of Object.keys(en.fields[`/filingStatusOptions`])) {
      const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
      const { factGraph } = setupFactGraph({
        ...baseFilerData,
        '/form1099Gs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [`${formId}`] },
        },
        [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(true),
        [`/filingStatus`]: createEnumWrapper(key, `/filingStatusOptions`),
      });
      const isMfj = key === `marriedFilingJointly`;
      it(`${key} and there is an 1099-G, it ${isMfj ? `asks` : `does not ask`} whose income is to be entered`, ({
        task,
      }) => {
        expect(givenFacts(factGraph).atPath(`${path}/add-1099-g`, formId, task)).toRouteNextTo(
          isMfj ? `${path}/1099-g-add-whose` : `${path}/1099-g-payer`
        );
      });
    }
  });

  it(`moves from 1099-G add whose to 1099-G payer`, ({ task }) => {
    const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/form1099Gs': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${formId}`] },
      },
      [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(true),
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
    });
    expect(givenFacts(factGraph).atPath(`${path}/1099-g-add-whose`, formId, task)).toRouteNextTo(
      `${path}/1099-g-payer`
    );
  });

  it(`moves from 1099-G payer to 1099-G amount`, ({ task }) => {
    const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/form1099Gs': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${formId}`] },
      },
      [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(true),
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
    });
    expect(givenFacts(factGraph).atPath(`${path}/1099-g-payer`, formId, task)).toRouteNextTo(`${path}/1099-g-amount`);
  });

  it(`moves from 1099-G amount to 1099-G repay options`, ({ task }) => {
    const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/form1099Gs': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${formId}`] },
      },
      [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(true),
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
      [`/form1099Gs/#${formId}/amount`]: createDollarWrapper(`20`),
    });
    expect(givenFacts(factGraph).atPath(`${path}/1099-g-amount`, formId, task)).toRouteNextTo(
      `${path}/1099-g-repay-options`
    );
  });

  it(`moves from 1099-G repay options to 1099-G repay amount`, ({ task }) => {
    const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/form1099Gs': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${formId}`] },
      },
      [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(true),
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
      [`/form1099Gs/#${formId}/amount`]: createDollarWrapper(`20`),
      [`/form1099Gs/#${formId}/repaid`]: createBooleanWrapper(true),
    });
    expect(givenFacts(factGraph).atPath(`${path}/1099-g-repay-options`, formId, task)).toRouteNextTo(
      `${path}/1099-g-repay-amount`
    );
  });

  it(`moves from 1099-G repay amount to 1099-G withholding`, ({ task }) => {
    const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/form1099Gs': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${formId}`] },
      },
      [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(true),
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
      [`/form1099Gs/#${formId}/amount`]: createDollarWrapper(`20`),
      [`/form1099Gs/#${formId}/repaid`]: createBooleanWrapper(true),
      [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsInTaxYear`]: createDollarWrapper(`30`),
      [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsBeforeTaxYear`]: createDollarWrapper(`40`),
    });
    expect(givenFacts(factGraph).atPath(`${path}/1099-g-repay-amount`, formId, task)).toRouteNextTo(
      `${path}/1099-g-withholding`
    );
  });

  it(`moves from 1099-G withholding to 1099-G payer tin if the value in box 4 is greater than 0`, ({ task }) => {
    const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/form1099Gs': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${formId}`] },
      },
      [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(true),
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
      [`/form1099Gs/#${formId}/amount`]: createDollarWrapper(`20`),
      [`/form1099Gs/#${formId}/repaid`]: createBooleanWrapper(true),
      [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsInTaxYear`]: createDollarWrapper(`30`),
      [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsBeforeTaxYear`]: createDollarWrapper(`40`),
      [`/form1099Gs/#${formId}/writableFederalTaxWithheld`]: createDollarWrapper(`10`),
    });
    expect(givenFacts(factGraph).atPath(`${path}/1099-g-withholding`, formId, task)).toRouteNextTo(
      `${path}/1099-g-payer-tin`
    );
  });

  it(`moves from 1099-G withholding to 1099-G state info if the value in box 4 is 0`, ({ task }) => {
    const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/form1099Gs': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${formId}`] },
      },
      [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(true),
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
      [`/form1099Gs/#${formId}/amount`]: createDollarWrapper(`20`),
      [`/form1099Gs/#${formId}/repaid`]: createBooleanWrapper(true),
      [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsInTaxYear`]: createDollarWrapper(`30`),
      [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsBeforeTaxYear`]: createDollarWrapper(`40`),
      [`/form1099Gs/#${formId}/writableFederalTaxWithheld`]: createDollarWrapper(`0`),
    });
    expect(givenFacts(factGraph).atPath(`${path}/1099-g-withholding`, formId, task)).toRouteNextTo(
      `${path}/1099-g-state-info`
    );
  });

  it(`moves from 1099-G payer tin to 1099-G state info`, ({ task }) => {
    const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/form1099Gs': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${formId}`] },
      },
      [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(true),
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
      [`/form1099Gs/#${formId}/amount`]: createDollarWrapper(`20`),
      [`/form1099Gs/#${formId}/repaid`]: createBooleanWrapper(true),
      [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsInTaxYear`]: createDollarWrapper(`30`),
      [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsBeforeTaxYear`]: createDollarWrapper(`40`),
      [`/form1099Gs/#${formId}/writableFederalTaxWithheld`]: createDollarWrapper(`10`),
      [`/form1099Gs/#${formId}/payer/tin`]: createEinWrapper(`55`, `5555555`),
    });
    expect(givenFacts(factGraph).atPath(`${path}/1099-g-payer-tin`, formId, task)).toRouteNextTo(
      `${path}/1099-g-state-info`
    );
  });

  it(`moves from 1099-G state info to data view of unemployment`, ({ task }) => {
    const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/form1099Gs': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${formId}`] },
      },
      [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(true),
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
      [`/form1099Gs/#${formId}/amount`]: createDollarWrapper(`20`),
      [`/form1099Gs/#${formId}/repaid`]: createBooleanWrapper(true),
      [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsInTaxYear`]: createDollarWrapper(`30`),
      [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsBeforeTaxYear`]: createDollarWrapper(`40`),
      [`/form1099Gs/#${formId}/writableFederalTaxWithheld`]: createDollarWrapper(`10`),
      [`/form1099Gs/#${formId}/payer/tin`]: createEinWrapper(`55`, `5555555`),
      [`/form1099Gs/#${formId}/writableState`]: createEnumWrapper(`sameState`, `/incomeFormStateOptions`),
      [`/form1099Gs/#${formId}/writableStateIdNumber`]: createStringWrapper(`12345`),
      [`/form1099Gs/#${formId}/writableStateTaxWithheld`]: createDollarWrapper(`10`),
    });
    expect(givenFacts(factGraph).atPath(`${path}/1099-g-state-info`, formId, task)).toRouteNextTo(
      `/data-view/loop/%2Fform1099Gs/${formId}`
    );
  });

  describe(`When filing status is ...`, () => {
    for (const key of Object.keys(en.fields[`/filingStatusOptions`])) {
      const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
      const { factGraph } = setupFactGraph({
        ...baseFilerData,
        '/form1099Gs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [`${formId}`] },
        },
        [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(false),
        [`/filingStatus`]: createEnumWrapper(key, `/filingStatusOptions`),
      });
      const isMfj = key === `marriedFilingJointly`;
      it(`${key} and there is no 1099-G, it ${isMfj ? `asks` : `does not ask`} whose income is to be entered`, ({
        task,
      }) => {
        expect(givenFacts(factGraph).atPath(`${path}/add-1099-g`, formId, task)).toRouteNextTo(
          isMfj ? `${path}/no-1099-g-add-whose` : `${path}/no-1099-g-payer-name`
        );
      });
    }
  });

  it(`moves from no 1099-G payer to no 1099-G amount`, ({ task }) => {
    const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/form1099Gs': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${formId}`] },
      },
      [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(false),
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
    });
    expect(givenFacts(factGraph).atPath(`${path}/no-1099-g-payer-name`, formId, task)).toRouteNextTo(
      `${path}/no-1099-g-amount`
    );
  });

  it(`moves from no 1099-G repay options to no 1099-G repay amount`, ({ task }) => {
    const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/form1099Gs': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${formId}`] },
      },
      [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(false),
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
      [`/form1099Gs/#${formId}/amount`]: createDollarWrapper(`20`),
      [`/form1099Gs/#${formId}/repaid`]: createBooleanWrapper(true),
    });
    expect(givenFacts(factGraph).atPath(`${path}/no-1099-g-repay-options`, formId, task)).toRouteNextTo(
      `${path}/no-1099-g-repay-amount`
    );
  });

  it(`moves from no 1099-G repay amount to data view of unemployment`, ({ task }) => {
    const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/form1099Gs': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${formId}`] },
      },
      [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(false),
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
      [`/form1099Gs/#${formId}/amount`]: createDollarWrapper(`20`),
      [`/form1099Gs/#${formId}/repaid`]: createBooleanWrapper(true),
      [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsInTaxYear`]: createDollarWrapper(`30`),
      [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsBeforeTaxYear`]: createDollarWrapper(`40`),
    });
    expect(givenFacts(factGraph).atPath(`${path}/no-1099-g-repay-amount`, formId, task)).toRouteNextTo(
      `/data-view/loop/%2Fform1099Gs/${formId}`
    );
  });

  describe(`knocks out the filer`, () => {
    it(`for when the repaid amount before the tax year is greater than 3000`, ({ task }) => {
      const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
      const { factGraph } = setupFactGraph({
        ...baseFilerData,
        '/form1099Gs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [`${formId}`] },
        },
        [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(true),
        [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
        [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
        [`/form1099Gs/#${formId}/amount`]: createDollarWrapper(`20`),
        [`/form1099Gs/#${formId}/repaid`]: createBooleanWrapper(true),
        [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsInTaxYear`]: createDollarWrapper(`30`),
        [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsBeforeTaxYear`]: createDollarWrapper(`3001`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-g-repay-amount`, formId, task)).toRouteNextTo(
        `${path}/1099-g-repay-knockout`
      );
    });

    it(`for having unemployment compensation in a different state`, ({ task }) => {
      const formId = `1a1e355e-3d19-415d-8470-fbafd9f58361`;
      const { factGraph } = setupFactGraph({
        ...baseFilerData,
        '/form1099Gs': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [`${formId}`] },
        },
        [`/form1099Gs/#${formId}/has1099`]: createBooleanWrapper(true),
        [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
        [`/form1099Gs/#${formId}/payer`]: createStringWrapper(`cat`),
        [`/form1099Gs/#${formId}/amount`]: createDollarWrapper(`20`),
        [`/form1099Gs/#${formId}/repaid`]: createBooleanWrapper(true),
        [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsInTaxYear`]: createDollarWrapper(`30`),
        [`/form1099Gs/#${formId}/writableAmountPaidBackForBenefitsBeforeTaxYear`]: createDollarWrapper(`40`),
        [`/form1099Gs/#${formId}/writableFederalTaxWithheld`]: createDollarWrapper(`10`),
        [`/form1099Gs/#${formId}/payer/tin`]: createEinWrapper(`55`, `5555555`),
        [`/form1099Gs/#${formId}/writableState`]: createEnumWrapper(`differentState`, `/incomeFormStateOptions`),
        [`/form1099Gs/#${formId}/writableStateIdNumber`]: createStringWrapper(`12345`),
        [`/form1099Gs/#${formId}/writableStateTaxWithheld`]: createDollarWrapper(`10`),
      });
      expect(givenFacts(factGraph).atPath(`${path}/1099-g-state-info`, formId, task)).toRouteNextTo(
        `${path}/1099-g-other-state-ko`
      );
    });
  });
});
