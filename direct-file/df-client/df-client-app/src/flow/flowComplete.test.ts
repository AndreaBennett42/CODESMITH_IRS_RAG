import flowNodes from './flow.js';
import { findFlowNodesOfType } from './findFlowNodesOfType.js';
import { SubcategoryNode } from './flowDeclarations.js';
import { generateDependencyGraph } from '../fact-dictionary/generate-src/dependencyGraph.js';

// Tests that all of the Subcategory completion conditions are a part of flow.xml:/flowIncomplete
// If this test fails, you probably need to update that fact because you changed a completeIf prop

// These conditions/facts are allowed in completeIf even without being a part of the incomplete fact
const ADDITIONAL_ALLOWED_CONDITIONS = [`/flowTrue`];
const SIGN_AND_SUBMIT_SUBCATEGORIES = [`review`, `sign-and-submit`, `submit`, `print-and-mail`];

const INCOMPLETE_PATH_FACT = `/flowIncomplete`;
const flowIncompleteDependencies = generateDependencyGraph()
  .get(INCOMPLETE_PATH_FACT)
  ?.map((condition) => condition.path);

if (flowIncompleteDependencies === undefined) {
  // Make the test failure condition clearer if the incomplete fact graph dependencies are missing
  it.failing(`Dependency list for ${INCOMPLETE_PATH_FACT} is missing`);
} else {
  // 'submissionBlockingFactsAreFalse' is a valid criteria even if it is not
  const allowableCompletionConditionPaths = [...flowIncompleteDependencies, ...ADDITIONAL_ALLOWED_CONDITIONS];
  describe(`Dependency condition`, () => {
    const subcategories: SubcategoryNode[] = findFlowNodesOfType(flowNodes, `Subcategory`);
    const subcategoryCompletionConditions = subcategories
      .map((subcategory) => [subcategory.props.route, subcategory.props.completeIf] as const)
      .filter(([route, completeIf]) => !SIGN_AND_SUBMIT_SUBCATEGORIES.includes(route) && completeIf !== undefined);

    const completionConditionPathsWithRoute = subcategoryCompletionConditions.flatMap(
      ([route, completionCondition]) => {
        const normalizedCondition = Array.isArray(completionCondition) ? completionCondition : [completionCondition];

        return normalizedCondition.map(
          (condition) =>
            // get path from condition
            [route, typeof condition === `string` ? condition : condition.condition] as const
        );
      }
    );

    test.each(completionConditionPathsWithRoute)(`for route %s is required for submission (%s)`, (_, conditionPath) => {
      expect(allowableCompletionConditionPaths).toEqual(expect.arrayContaining([conditionPath]));
    });
  });
}
