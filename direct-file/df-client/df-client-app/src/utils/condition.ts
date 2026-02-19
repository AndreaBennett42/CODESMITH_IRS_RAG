import { FactGraph } from '@irs/js-factgraph-scala';
import { Condition, RawCondition } from '../flow/Condition.js';
type HasConditions = { condition?: RawCondition; conditions?: RawCondition[] };

export const conditionsPass = (props: HasConditions, fg: FactGraph, collectionId: string | null) => {
  if (props && props.condition) {
    return new Condition(props.condition).evaluate(fg, collectionId);
    // eslint-disable-next-line eqeqeq
  } else if (props.conditions != undefined) {
    return props.conditions.every((c) => {
      const result = new Condition(c).evaluate(fg, collectionId);
      return result;
    });
  } else {
    return true;
  }
};
