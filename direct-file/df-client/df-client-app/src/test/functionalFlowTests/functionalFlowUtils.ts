import { Task, expect } from 'vitest';
import { FlowConfig } from '../../flow/flowConfig.js';
import getNextScreen, { GetNextScreenOpts } from '../../screens/getNextScreen.js';

import type { FactGraph } from '@irs/js-factgraph-scala';
import { ScreenConfig } from '../../flow/ScreenConfig.js';

export type FunctionalGiven = {
  screen: ScreenConfig;
  factGraph: FactGraph;
  collectionId: null | string;
  flow: FlowConfig;
};

export default function makeGivenFacts(flow: FlowConfig) {
  return function givenFacts(factGraph: FactGraph): {
    atPath: (route: string, collectionId: null | string, task: Task) => FunctionalGiven;
  } {
    return {
      atPath: (route: string, collectionId: null | string, task: Task): FunctionalGiven => {
        const screen = flow.screensByRoute.get(route);
        if (!task.meta.functionalFlowTestStartingScreenRoutes) {
          task.meta.functionalFlowTestStartingScreenRoutes = new Set();
        }
        task.meta.functionalFlowTestStartingScreenRoutes.add(route);
        if (!screen) {
          throw new Error(`${route} does not exist in flow`);
        }
        return { screen, factGraph, collectionId, flow };
      },
    };
  };
}

expect.extend({
  toRouteNextTo(received: FunctionalGiven, route: string, additionalMsg?: string, nextScreenOpts?: GetNextScreenOpts) {
    const nextScreen = getNextScreen(
      received.screen,
      received.factGraph,
      received.collectionId,
      received.flow,
      nextScreenOpts
    );
    const nextPath = nextScreen.routable.fullRoute(nextScreen.collectionId);

    const expectScreen = received.flow.screensByRoute.get(route);
    const expectedPath = expectScreen !== undefined ? expectScreen.fullRoute(nextScreen.collectionId) : route;
    const baseMessage = `${nextPath} is not ${expectedPath}`;
    return {
      pass: nextPath === expectedPath,
      message: () => (additionalMsg ? `${baseMessage}\n${additionalMsg}` : baseMessage),
    };
  },
});
