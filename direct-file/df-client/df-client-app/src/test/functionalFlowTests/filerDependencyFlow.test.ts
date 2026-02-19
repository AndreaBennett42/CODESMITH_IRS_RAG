import { it, describe, expect } from 'vitest';
import { baseFilerData, primaryFilerId } from '../testData.js';
import { createBooleanWrapper } from '../persistenceWrappers.js';
import { createFlowConfig } from '../../flow/flowConfig.js';
import flowNodes from '../../flow/flow.js';
import makeGivenFacts from './functionalFlowUtils.js';
import { setupFactGraph } from '../setupFactGraph.js';

const flow = createFlowConfig(flowNodes);
const givenFacts = makeGivenFacts(flow);

describe(`The "claim" section of "about you"`, () => {
  it(`skips all claim questions if canBeClaimed is "no"`, ({ task }) => {
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
    });
    expect(
      givenFacts(factGraph).atPath(
        `/flow/you-and-your-family/about-you/about-you-could-be-claimed`,
        primaryFilerId,
        task
      )
    ).toRouteNextTo(`/data-view/flow/you-and-your-family/about-you`);
  });

  it(`moves to about-you-claimer-filing-requirement if about-you-could-be-claimed is true`, ({ task }) => {
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
    });
    expect(
      givenFacts(factGraph).atPath(
        `/flow/you-and-your-family/about-you/about-you-could-be-claimed`,
        primaryFilerId,
        task
      )
    ).toRouteNextTo(
      `/flow/you-and-your-family/about-you/about-you-claimer-filing-requirement?%2FprimaryFiler=${primaryFilerId}`
    );
  });
  it(`moves to about-you-claimer-filing if about-you-claimer-filing-requirement is "no"`, ({ task }) => {
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
    });

    expect(
      givenFacts(factGraph).atPath(
        `/flow/you-and-your-family/about-you/about-you-claimer-filing-requirement`,
        primaryFilerId,
        task
      )
    ).toRouteNextTo(`/flow/you-and-your-family/about-you/about-you-claimer-filing?%2FprimaryFiler=${primaryFilerId}`);
  });
  it(`moves to about-you-will-be-claimed if about-you-claimer-filing-requirement is "yes"`, ({ task }) => {
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),

      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
    });
    expect(
      givenFacts(factGraph).atPath(
        `/flow/you-and-your-family/about-you/about-you-claimer-filing-requirement`,
        primaryFilerId,
        task
      )
    ).toRouteNextTo(`/flow/you-and-your-family/about-you/about-you-will-be-claimed`);
  });

  it(`moves to about-you-claimers-return if about-you-claimer-filing is yes,`, ({ task }) => {
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/potentialClaimerDidFile`]: createBooleanWrapper(true),
    });

    expect(
      givenFacts(factGraph).atPath(`/flow/you-and-your-family/about-you/about-you-claimer-filing`, primaryFilerId, task)
    ).toRouteNextTo(`/flow/you-and-your-family/about-you/about-you-claimers-return?%2FprimaryFiler=${primaryFilerId}`);
  });

  it(`skips all other questions in the section if about-you-claimer-filing is "no"`, ({ task }) => {
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/potentialClaimerDidFile`]: createBooleanWrapper(false),
    });
    expect(
      givenFacts(factGraph).atPath(`/flow/you-and-your-family/about-you/about-you-claimer-filing`, primaryFilerId, task)
    ).toRouteNextTo(`/data-view/flow/you-and-your-family/about-you`);
  });

  it(`moves to about-you-will-be-claimed if the answer to about-you-claimers-return is "no"`, ({ task }) => {
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/potentialClaimerDidFile`]: createBooleanWrapper(true),
      [`/primaryFilerPotentialClaimerFiledOnlyForRefund`]: createBooleanWrapper(false),
    });

    expect(
      givenFacts(factGraph).atPath(
        `/flow/you-and-your-family/about-you/about-you-claimers-return`,
        primaryFilerId,
        task
      )
    ).toRouteNextTo(`/flow/you-and-your-family/about-you/about-you-will-be-claimed`);
  });

  it(`moves to checklist if the answer to about-you-claimers-return is "yes"`, ({ task }) => {
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/potentialClaimerDidFile`]: createBooleanWrapper(true),
      [`/primaryFilerPotentialClaimerFiledOnlyForRefund`]: createBooleanWrapper(true),
    });

    expect(
      givenFacts(factGraph).atPath(
        `/flow/you-and-your-family/about-you/about-you-claimers-return`,
        primaryFilerId,
        task
      )
    ).toRouteNextTo(`/data-view/flow/you-and-your-family/about-you`);
  });

  it(`moves to checklist regardless of the answer to about-you-will-be-claimed`, ({ task }) => {
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/potentialClaimerDidFile`]: createBooleanWrapper(true),
      [`/primaryFilerPotentialClaimerFiledOnlyForRefund`]: createBooleanWrapper(false),
      [`/filers/#${primaryFilerId}/willBeClaimed`]: createBooleanWrapper(true),
    });

    expect(
      givenFacts(factGraph).atPath(
        `/flow/you-and-your-family/about-you/about-you-will-be-claimed`,
        primaryFilerId,
        task
      )
    ).toRouteNextTo(`/data-view/flow/you-and-your-family/about-you`);
  });
});
