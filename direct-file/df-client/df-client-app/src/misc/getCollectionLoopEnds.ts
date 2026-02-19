import { FactGraph } from '@irs/js-factgraph-scala';
import { FlowConfig } from '../flow/flowConfig.js';
import getNextScreen, { NextScreenData } from '../screens/getNextScreen.js';
import { ScreenConfig } from '../flow/ScreenConfig.js';
import { findLast } from '../utils/polyfills.js';

export const getFirstAvailableOfCollectionLoop = (
  loopName: string,
  factGraph: FactGraph,
  collectionId: string | null,
  flow: FlowConfig
): NextScreenData => {
  const loop = flow.collectionLoopsByName.get(loopName);
  if (!loop) throw new Error(`Cannot find start of loop`);

  const firstScreen = loop.screens[0];

  if (firstScreen?.isAvailable(factGraph, collectionId)) {
    return { routable: firstScreen, collectionId };
  }

  return getNextScreen(firstScreen, factGraph, collectionId, flow);
};

export const getEndOfCollectionLoop = (
  loopName: string,
  factGraph: FactGraph,
  collectionId: string | null,
  flow: FlowConfig,
  isAvailable: boolean
): ScreenConfig | undefined => {
  const loop = flow.collectionLoopsByName.get(loopName);
  if (!loop) throw new Error(`Cannot find start of loop`);

  // We find the last screen to find the end of the loop.
  if (isAvailable) {
    // if isAvailable is true, we want to find the last screen that is available
    return findLast(loop.screens, (screen) => screen.isAvailable(factGraph, collectionId));
  } else {
    return loop.screens[loop.screens.length - 1];
  }
};
