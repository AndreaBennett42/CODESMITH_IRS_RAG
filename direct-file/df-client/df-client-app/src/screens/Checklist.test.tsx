import { it, describe, expect } from 'vitest';
import { createFlowConfig } from '../flow/flowConfig.js';
import flowNodes from '../flow/flow.js';
import {
  baseFilerData,
  filerWithRefundDueData,
  filerWithPaymentDueData,
  filerWithZeroBalanceData,
  primaryFilerId,
} from '../test/testData.js';
import { createBooleanWrapper, createEnumWrapper } from '../test/persistenceWrappers.js';
import { Path } from '../flow/Path.js';
import { getChecklistState } from '../flow/useChecklistState.js';
import { setupFactGraph } from '../test/setupFactGraph.js';

describe(`Checklist state`, () => {
  it(`Begins with the first category expanded and the first subcategory active`, () => {
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph();
    const categories = getChecklistState(factGraph, flow);
    expect(categories[0].categoryActive).toBe(true);
    expect(categories[0].subcategories[0].isComplete).toBe(false);
    expect(categories[0].subcategories[0].isNext).toBe(true);
    const otherCategories = categories.slice(1);
    for (const cat of otherCategories) {
      expect(cat.categoryActive).toBe(false);
    }
  });

  it(`Marks subcategories as complete when their completion condition is true`, () => {
    // This test is dependent on "/flowPrimaryFilerDependentsTest" being the
    // completion condition for the first section in the flow
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
      '/filerResidenceAndIncomeState': createEnumWrapper(`ny`, `/scopedStateoptions`),
      [`/primaryFilerW2And1099IntInScopedState`]: createEnumWrapper(
        `onlySame`,
        `/primaryFilerW2And1099IntStateOptions`
      ),
      [`/receivedAlaskaPfd`]: createBooleanWrapper(false),
    });
    const subcategoryComplete = factGraph.get(
      Path.concretePath(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`, primaryFilerId)
    );
    expect(subcategoryComplete.complete).toBe(true);
    const categories = getChecklistState(factGraph, flow);
    expect(categories[0].categoryActive).toBe(true);
    expect(categories[0].subcategories[0].isComplete).toBe(true);
    expect(categories[0].subcategories[0].isNext).toBe(false);
  });

  it(`Activates the next subcategory when the previous subcategory is complete`, () => {
    // This test is dependent on "/flowPrimaryFilerDependentsTest" being the
    // completion condition for the first section in the flow
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
      '/filerResidenceAndIncomeState': createEnumWrapper(`ny`, `/scopedStateoptions`),
      [`/primaryFilerW2And1099IntInScopedState`]: createEnumWrapper(
        `onlySame`,
        `/primaryFilerW2And1099IntStateOptions`
      ),
      [`/receivedAlaskaPfd`]: createBooleanWrapper(false),
    });
    const subcategoryComplete = factGraph.get(
      Path.concretePath(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`, primaryFilerId)
    );
    expect(subcategoryComplete.complete).toBe(true);
    const categories = getChecklistState(factGraph, flow);
    expect(categories[0].categoryActive).toBe(true);
    expect(categories[0].subcategories[0].isComplete).toBe(true);
    expect(categories[0].subcategories[0].isNext).toBe(false);
    expect(categories[0].subcategories[1].isNext).toBe(true);
    expect(categories[0].subcategories[1].isComplete).toBe(false);
  });

  it(`Expands a category when a subcategory is complete`, () => {
    // This test is dependent on "/flowPrimaryFilerDependentsTest" being the
    // completion condition for the first section in the flow
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
    });
    const subcategoryComplete = factGraph.get(
      Path.concretePath(`/primaryFilerCouldBeDependentAndClaimerIsRequiredToFileOrFilingForCredits`, primaryFilerId)
    );
    expect(subcategoryComplete.complete).toBe(true);
    const categories = getChecklistState(factGraph, flow);
    expect(categories[0].categoryActive).toBe(true);
  });

  it(`Provides a first route to a subcategory only when the category is active and screens are available`, () => {
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph();
    const categories = getChecklistState(factGraph, flow);
    for (const cat of categories) {
      if (cat.categoryActive) {
        for (const sc of cat.subcategories) {
          expect(sc.navigationUrl).toBeDefined();
        }
      } else {
        for (const sc of cat.subcategories) {
          expect(sc.navigationUrl).toBeUndefined();
        }
      }
    }
  });

  it(`returns a data view if there is a data view for that subsection`, () => {
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
      '/filerResidenceAndIncomeState': createEnumWrapper(`ny`, `/scopedStateoptions`),
      [`/primaryFilerW2And1099IntInScopedState`]: createEnumWrapper(
        `onlySame`,
        `/primaryFilerW2And1099IntStateOptions`
      ),
      [`/receivedAlaskaPfd`]: createBooleanWrapper(false),
    });
    const categories = getChecklistState(factGraph, flow);
    for (const cat of categories) {
      if (cat.categoryActive) {
        for (const sc of cat.subcategories) {
          if (sc.isComplete && flow.subcategoriesByRoute.get(sc.subcategoryRoute)?.hasDataView) {
            expect(sc.navigationUrl).toMatch(/^\/data-view/);
          } else {
            expect(sc.navigationUrl).toMatch(/^\/flow/);
          }
        }
      }
    }
  });

  it(`Hides the payment method section when the filer has zero balance`, () => {
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph(filerWithZeroBalanceData);
    const categories = getChecklistState(factGraph, flow);
    // eslint-disable-next-line eqeqeq
    const yourTaxesSection = categories.find((cat) => cat.route == `/flow/your-taxes`);
    const paymentSubsection = yourTaxesSection?.subcategories.find(
      (sc) => sc.subcategoryRoute === `/flow/your-taxes/payment-method`
    );
    expect(paymentSubsection).toBeUndefined();
  });

  it(`Shows the payment method section when the filer has a balance to pay`, () => {
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph(filerWithPaymentDueData);
    const categories = getChecklistState(factGraph, flow);
    // eslint-disable-next-line eqeqeq
    const yourTaxesSection = categories.find((cat) => cat.route == `/flow/your-taxes`);
    const paymentSubsection = yourTaxesSection?.subcategories.find(
      (sc) => sc.subcategoryRoute === `/flow/your-taxes/payment-method`
    );
    expect(paymentSubsection).toBeDefined();
  });

  it(`Shows the payment method section when the filer has a refund`, () => {
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph(filerWithRefundDueData);
    const categories = getChecklistState(factGraph, flow);
    // eslint-disable-next-line eqeqeq
    const yourTaxesSection = categories.find((cat) => cat.route == `/flow/your-taxes`);
    const paymentSubsection = yourTaxesSection?.subcategories.find(
      (sc) => sc.subcategoryRoute === `/flow/your-taxes/payment-method`
    );
    expect(paymentSubsection).toBeDefined();
  });

  it(`Shows the 1099R pre-release section when the feature flag has not been enabled`, () => {
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/is1099RFeatureFlagEnabled': createBooleanWrapper(false),
    });
    const categories = getChecklistState(factGraph, flow);
    // eslint-disable-next-line eqeqeq
    const incomeSection = categories.find((cat) => cat.route == `/flow/income`);
    const preReleaseSubsection = incomeSection?.subcategories.find(
      (sc) => sc.subcategoryRoute === `/flow/income/retirement-launch`
    );
    expect(preReleaseSubsection).toBeDefined();
  });
  it(`Hides the 1099R pre-release section when the feature flag has been enabled`, () => {
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/is1099RFeatureFlagEnabled': createBooleanWrapper(true),
    });
    const categories = getChecklistState(factGraph, flow);
    // eslint-disable-next-line eqeqeq
    const incomeSection = categories.find((cat) => cat.route == `/flow/income`);
    const preReleaseSubsection = incomeSection?.subcategories.find(
      (sc) => sc.subcategoryRoute === `/flow/income/retirement-launch`
    );
    expect(preReleaseSubsection).toBeUndefined();
  });
  it(`Shows the post-launch 1099R section when the feature flag has been enabled`, () => {
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/is1099RFeatureFlagEnabled': createBooleanWrapper(true),
    });
    const categories = getChecklistState(factGraph, flow);
    // eslint-disable-next-line eqeqeq
    const incomeSection = categories.find((cat) => cat.route == `/flow/income`);
    const postLaunchSubsection = incomeSection?.subcategories.find(
      (sc) => sc.subcategoryRoute === `/flow/income/retirement`
    );
    expect(postLaunchSubsection).toBeDefined();
  });
  it(`Hides the post-launch 1099R section when the feature flag has not been enabled`, () => {
    const flow = createFlowConfig(flowNodes);
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/is1099RFeatureFlagEnabled': createBooleanWrapper(false),
    });
    const categories = getChecklistState(factGraph, flow);
    // eslint-disable-next-line eqeqeq
    const incomeSection = categories.find((cat) => cat.route == `/flow/income`);
    const postLaunchSubsection = incomeSection?.subcategories.find(
      (sc) => sc.subcategoryRoute === `/flow/income/retirement`
    );
    expect(postLaunchSubsection).toBeUndefined();
  });
});
