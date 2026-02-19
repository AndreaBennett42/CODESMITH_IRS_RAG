export function findFlowNodesWithProp(
  el: JSX.Element,
  propName:
    | 'route'
    | 'i18nKey'
    | 'path'
    | 'condition'
    | 'name'
    | 'loopName'
    | 'required'
    | `editRoute`
    | `knockoutRoute`,
  propValue: string | null | boolean = null
) {
  const queue: JSX.Element[] = [el];
  const ret: JSX.Element[] = [];
  // eslint-disable-next-line eqeqeq
  while (queue.length != 0) {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const testEl = queue.pop()!;
    // Check if the element's property equals propValue
    if (testEl?.props && (propValue === null || testEl.props[propName] === propValue)) {
      ret.push(testEl);
    }
    // Push child nodes onto the end of the queue
    if (testEl && testEl.props) {
      queue.push(...(Array.isArray(testEl.props.children) ? testEl.props.children : [testEl.props.children]));
    }
  }
  return ret;
}
