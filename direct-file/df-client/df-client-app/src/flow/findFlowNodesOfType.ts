import type { AllComponentTypes } from './ContentDeclarations.js';
import type { AllFlowTypes } from './flowDeclarations.js';

type AnyFlowNode = keyof typeof AllComponentTypes | keyof typeof AllFlowTypes;

export function findFlowNodesOfType(el: JSX.Element, type: AnyFlowNode) {
  const queue: JSX.Element[] = [el];
  const ret: JSX.Element[] = [];
  // eslint-disable-next-line eqeqeq
  while (queue.length != 0) {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const testEl = queue.pop()!;
    if (testEl?.type?.name === type) {
      ret.push(testEl);
    }
    if (testEl && testEl.props) {
      queue.push(...(Array.isArray(testEl.props.children) ? testEl.props.children : [testEl.props.children]));
    }
  }
  return ret;
}
