import { findFlowNodesOfType } from './findFlowNodesOfType.js';
import { findFlowNodesWithProp } from './findFlowNodesWithProp.js';
import flowNodes from './flow.js';

/**
 * Validate that there are no semantically invalid combinations of props.
 *
 * These tests check for invalid node props that can't be solved with typing
 */
describe(`Flow nodes`, () => {
  it(`has no mixed condition declarations`, () => {
    // NOTE: This was previously prevented by making BaseContentDeclaration a discriminated union
    //    unfortunately, the base union made additional typing improvements difficult
    const conditionalNodes = findFlowNodesWithProp(flowNodes, `condition`);

    const invalidNodes = conditionalNodes.filter((node) => node.props.condition && node.props.conditions?.length > 0);

    expect(invalidNodes).toEqual([]);
  });

  it(`has exactly one SectionReview`, () => {
    const sectionReviewNodes = findFlowNodesOfType(flowNodes, `SectionReview`);

    expect(sectionReviewNodes.length).toBe(1);
  });
});
