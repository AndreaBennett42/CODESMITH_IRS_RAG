import { FlowCollectionLoop, FlowSubcategory, FlowSubSubcategory } from '../flow/flowConfig.js';
import { AlertConfig } from './aggregatedAlertHelpers.js';
import { ScreenConfig } from '../flow/ScreenConfig.js';

function isLoop(value: FlowCollectionLoop | FlowSubcategory | FlowSubSubcategory): value is FlowCollectionLoop {
  // logic to determine if value is a FlowCollectionLoop
  return (value as FlowCollectionLoop)?.loopName !== undefined;
}

// Get the dataview section route for a screen
// TODO autoiterating screens are not working file ticket.
export const getSectionRouteForScreen = (
  screen: ScreenConfig,
  subcategoryOrLoop: FlowCollectionLoop | FlowSubcategory
) => {
  // In most cases we can just use the screen's subsubcategory route
  // as the subcategory will be displayed and we can link to it
  let parentRoute = screen.subSubcategoryRoute;
  if (screen.collectionLoop?.isInner && !isLoop(subcategoryOrLoop)) {
    // Except in the case of a nested collection hub
    parentRoute = screen.collectionLoop.fullRoute;
  }
  return parentRoute;
};

export const getSectionIndexForScreen = (
  sectionRoute: string,
  subSubcategoriesAndLoops: Array<FlowCollectionLoop | FlowSubSubcategory>
) => {
  const index = subSubcategoriesAndLoops.findIndex((sscOrLoop) => {
    if (isLoop(sscOrLoop)) {
      // Check the subsubcategory route
      return sscOrLoop.subSubcategories.find((ssc) => ssc.fullRoute === sectionRoute);
    }
    return sscOrLoop.fullRoute === sectionRoute;
  });
  return index;
};

/**
 * Get the screen that contains the collection item manager for an inner loop
 */
export const getInnerLoopHub = (ssc: FlowSubSubcategory) => {
  return ssc.screens.find((s: ScreenConfig) => {
    return s.content.find((contentItem) => {
      return contentItem.componentName === `CollectionItemManager`;
    });
  });
};

export const getLoopRoutes = (
  route: string,
  loopName: string,
  subcategoryOrLoop: FlowSubcategory | FlowCollectionLoop
) => {
  let sscRoute: string | undefined;
  let loopRoute: string | undefined;
  if (isLoop(subcategoryOrLoop)) {
    // this means we are at the collection item dataview
    sscRoute = subcategoryOrLoop.screens.find((s: ScreenConfig) => s.screenRoute === route)?.subSubcategoryRoute;
  } else {
    // this means we are at the normal dataview (but error is in a loop)
    const collectionLoop = subcategoryOrLoop.loops.find((l: FlowCollectionLoop) => l.loopName === loopName);
    loopRoute = collectionLoop?.fullRoute;
  }
  return { sscRoute, loopRoute };
};
// We only enter this function if the error was located in a loop
export const getSscForLoop = (e: AlertConfig, subcategoryOrLoop: FlowSubcategory | FlowCollectionLoop) => {
  // In most cases we can just use the screen's subsubcategory route
  // as the subcategory will be displayed and we can link to it
  // ONLY called if error has a loop attached.
  // Could be inner loopor not.
  // Cannot be at hub level, could be dataview with nested or autoiterated or colelction item
  // if nested
  let parentRoute: string | undefined;
  if (isLoop(subcategoryOrLoop)) {
    // this means we are at the collection item dataview
    parentRoute = e.subSubcategoryRoute;
  } else {
    // this means we are at the normal dataview (but error is in a loop)
    const collectionLoop = subcategoryOrLoop.loops.find((l: FlowCollectionLoop) => l.loopName === e.loopName);
    parentRoute = collectionLoop?.fullRoute;
  }
  const result = subcategoryOrLoop.subSubcategories.find((ssc) => {
    return ssc.fullRoute === parentRoute;
  });
  return result;
};
