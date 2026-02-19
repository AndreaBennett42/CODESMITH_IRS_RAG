import flowNodes from './flow.js';
import { FactDeclaration } from './ContentDeclarations.js';
import { findFlowNodesWithProp } from './findFlowNodesWithProp.js';
/**
Checkbox booleans are now required by default. Optional checkboxes that 
existed beforethe change must retain their current naming convention 
since making it a writable fact will not break existing user's returns.
*/
const EXISTING_OPTIONAL_CHECKBOX_FACTS = [`thirdPartySickPay`, `deceased`, `retirementPlan`, `statutoryEmployee`];

describe(`All optional nodes use facts prefixed with writable`, () => {
  const optionalNodes = findFlowNodesWithProp(flowNodes, `required`, false);
  for (const node of optionalNodes) {
    const props = node.props as FactDeclaration;
    it(`Optional node ${node.props.path} prefixed with writable`, () => {
      const factName = props.path.split(`/`).pop();
      // Skip the prefix check for existing optional checkboxes.
      if (EXISTING_OPTIONAL_CHECKBOX_FACTS.includes(factName as string)) {
        return;
      }
      try {
        expect(factName).toMatch(/^writable/);
      } catch (error) {
        throw new Error(
          `Looks like you tried to make a ${node.type.name} field optional. 
          For a field to be optional, you need to create a derived fact that handles 
          an incomplete writable fact and returns a default value. 
          Then prefix the writable fact with "writable".`
        );
      }
      expect(factName).toMatch(/^writable/);
    });
  }
});
