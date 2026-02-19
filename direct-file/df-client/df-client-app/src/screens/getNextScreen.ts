import { ConcretePath, FactGraph, scalaListToJsArray } from '@irs/js-factgraph-scala';
import { getEndOfCollectionLoop, getFirstAvailableOfCollectionLoop } from '../misc/getCollectionLoopEnds.js';
import { FlowConfig } from '../flow/flowConfig.js';
import { RouteOptions, ScreenConfig } from '../flow/ScreenConfig.js';

export interface GetNextScreenOpts {
  navigateToDataViewAtEndOfSubSubCategory?: boolean;
}

export const DEFAULT_OPTS: GetNextScreenOpts = {
  navigateToDataViewAtEndOfSubSubCategory: false,
};

export interface Routable {
  fullRoute: (collectionId: string | null, opts?: RouteOptions) => string;
}

function makeRoutable(route: string): Routable {
  return {
    fullRoute: () => route,
  };
}

function makeLoopRoute(route: string): Routable {
  return {
    fullRoute: (collectionId: string | null) => `${route}/${collectionId ?? undefined}`,
  };
}

export interface NextScreenData {
  routable: ScreenConfig | Routable;
  collectionId: string | null;
}

// Returns the first available screen and its collectionId in the list of screens provided
// This only uses the automatically routed order of screens provided and doesn't redirect to dataviews
// or checklists as those scenarios are handled in the getNextScreen function
const getNextScreenWithId = (
  currentScreenInfo: ScreenConfig,
  collectionId: string | null,
  factGraph: FactGraph,
  screens: ScreenConfig[]
): { nextScreen: ScreenConfig | null; nextCollectionId: string | null } => {
  // Given a collection, get the first item id
  const getFirstCollectionIdOfLoop = (collectionName: ConcretePath, factGraph: FactGraph) => {
    const result = factGraph.get(collectionName);
    const collectionItems = result.complete
      ? (scalaListToJsArray(result.get.getItemsAsStrings()) as string[])
      : undefined;
    return collectionItems ? collectionItems[0] : null;
  };

  // We loop through the screens checking conditions with the correct collectionId
  // and finally return the first available screen and its collectionId
  let prevScreenInfo = currentScreenInfo;
  let prevCollectionId = collectionId;
  for (const screenInfo of screens) {
    let maybeCollectionId = prevCollectionId;

    // If the screen is changing collection loops, clear the collectionId
    // eslint-disable-next-line eqeqeq
    if (!(prevScreenInfo.collectionLoop == screenInfo.collectionLoop)) {
      maybeCollectionId = null;
    }

    // If the screen is in a new auto-iterating collection loop we must use the first items id
    if (
      screenInfo.collectionLoop &&
      screenInfo.collectionLoop.autoIterate &&
      // eslint-disable-next-line eqeqeq
      (prevCollectionId == null || prevScreenInfo.collectionLoop !== screenInfo.collectionLoop)
    ) {
      maybeCollectionId = getFirstCollectionIdOfLoop(screenInfo.collectionContext as ConcretePath, factGraph);
    }

    prevScreenInfo = screenInfo;
    prevCollectionId = maybeCollectionId;
    // If the current screen is one that we automatically route to and it's available, return it.
    if (screenInfo.routeAutomatically && screenInfo.isAvailable(factGraph, maybeCollectionId)) {
      return { nextScreen: screenInfo, nextCollectionId: maybeCollectionId };
    }
  }
  return { nextScreen: null, nextCollectionId: null };
};

const getNextScreen = (
  currentScreenInfo: ScreenConfig,
  factGraph: FactGraph,
  collectionId: string | null,
  flow: FlowConfig,
  opts?: GetNextScreenOpts
): NextScreenData => {
  const options = { DEFAULT_OPTS, ...opts };
  const allScreens = flow.screensByRoute;
  // Get our current index in the list of screens
  const currentScreenIndex = flow.screens.findIndex((value) => value === currentScreenInfo);
  if (!currentScreenInfo || currentScreenIndex === -1) {
    throw new Error(`${currentScreenInfo} does not exist in all screens`);
  }

  // Create an array of screens after this one
  const startIndex = currentScreenIndex + 1;
  const screens = Array.from(allScreens.values()).slice(startIndex);
  const { collectionLoop } = currentScreenInfo;

  // HANDLE END OF COLLECTION LOOP
  const atEndOfCollectionLoop =
    collectionLoop &&
    collectionId &&
    // eslint-disable-next-line eqeqeq
    getEndOfCollectionLoop(collectionLoop.loopName, factGraph, collectionId, flow, true)?.screenRoute ==
      currentScreenInfo.screenRoute;

  // If at the last screen of an auto-iterating collection loop, iterate to the next item in the collection
  // (if available) and restart the loop instead of going to the next screen
  if (atEndOfCollectionLoop && collectionLoop.autoIterate && collectionId) {
    const collectionName = currentScreenInfo.collectionContext as ConcretePath;
    const result = factGraph.get(collectionName);
    const collectionItems = result.complete ? (scalaListToJsArray(result.get.getItemsAsStrings()) as string[]) : [];
    const currentIndex = collectionItems.indexOf(collectionId);
    if (currentIndex + 1 < collectionItems.length) {
      const maybeCollectionId = collectionItems[currentIndex + 1];
      // Because we passed the maybe collection id to get the first available of the collection loop, it is already
      // fully qualified with url params, and we can skip the rest of the function.
      return getFirstAvailableOfCollectionLoop(collectionName, factGraph, maybeCollectionId, flow);
    }
  }

  // If at the last screen of a non-auto-iterate collection loop, go to current collection item
  // data view after we complete each iteration of a loop
  if (atEndOfCollectionLoop && !collectionLoop.autoIterate && collectionId) {
    return {
      routable: makeLoopRoute(`/data-view/loop/${encodeURIComponent(collectionLoop.loopName)}`),
      collectionId: collectionId,
    };
  }

  // Get the next available screen and its collectionId
  const { nextScreen, nextCollectionId } = getNextScreenWithId(currentScreenInfo, collectionId, factGraph, screens);
  if (!nextScreen) {
    throw new Error(`No available next route`);
  }
  const hasSameSubcategory = currentScreenInfo.subcategoryRoute === nextScreen.subcategoryRoute;
  const hasSameSubSubcategory = currentScreenInfo.subSubcategoryRoute === nextScreen.subSubcategoryRoute;

  // Knockout case: Go to knockout!
  if (nextScreen.isKnockout) {
    return { routable: nextScreen, collectionId: nextCollectionId };
  } else if (options.navigateToDataViewAtEndOfSubSubCategory && !hasSameSubSubcategory) {
    // Next screen is
    //   - in the same subsubcategory AND
    //   - therefore in the same subcategory AND
    //   - review mode is true

    // Act as Dataview case: Some subsubcategories have a special dataview, if so, go back to checklist!
    if (currentScreenInfo.actAsDataView) {
      return { routable: makeRoutable(`/checklist`), collectionId };
    }

    // Split category case: We may be in a subsubcategory that's divided with on or more other subsubcategories
    // in between
    // <SubSubCategory route="a" />
    // <SubSubCategory route="b" />
    // <SubSubCategory route="a" />
    // This PR has more details: https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/4042

    // We have to check the subcategory object to see if there are any screens left in the same subsubcategory.
    const subSubcategory = flow.subsubcategoriesByRoute.get(currentScreenInfo.subSubcategoryRoute);
    if (!subSubcategory) {
      throw new Error(`Screen ${currentScreenInfo.route} isn't in a subsubcategory?`);
    }
    const indexInSubSubcategory = subSubcategory.screens.findIndex((sc) => sc.route === currentScreenInfo.route);
    const nextScreensInSubSubcategory = subSubcategory.screens.slice(indexInSubSubcategory + 1);

    const { nextScreen: nextScreenInSubSubcategory, nextCollectionId } = getNextScreenWithId(
      currentScreenInfo,
      collectionId,
      factGraph,
      nextScreensInSubSubcategory
    );

    // Next screen is available in the same subsubcategory, go there!
    if (nextScreenInSubSubcategory) {
      return { routable: nextScreenInSubSubcategory, collectionId: nextCollectionId };
    }

    // Otherwise, go to the data view for this subsubcategory with current collectionId
    return {
      routable:
        getDataViewRoute(collectionLoop, collectionId, flow, currentScreenInfo, true) || makeRoutable(`/checklist`),
      collectionId,
    };
  } else if (hasSameSubcategory) {
    // Next screen is in the same subcategory AND
    //   - either we're not in review mode OR
    //   - next screen is within the same subsubcategory
    // we go to next screen.
    return { routable: nextScreen, collectionId: nextCollectionId };
  } else {
    // Next screen is
    //  - in a new subcategory AND
    //  - therefore also new subsubcategory
    const subcategory = flow.subcategoriesByRoute.get(currentScreenInfo.subcategoryRoute);

    // Check if the current subcategory has a screen acting as the data view
    const hasScreenActingAsDataView = subcategory?.screens.find((sc) => sc.actAsDataView);
    // If the subcategory:
    // has a screen acting as a data view,
    // or does NOT have a data view,
    // or the subcategory has a collection hub, we can go back to the checklist.
    // Otherwise, go to the data view.
    const subcategoryHasCollectionWithLoops = subcategory?.collectionName && subcategory?.loops.length > 0;
    const goToChecklist = hasScreenActingAsDataView || !subcategory?.hasDataView || subcategoryHasCollectionWithLoops;
    const route = goToChecklist
      ? makeRoutable(`/checklist`)
      : getDataViewRoute(collectionLoop, collectionId, flow, currentScreenInfo, false);

    return {
      routable: route,
      collectionId,
    };
  }
};

const getDataViewRoute = (
  collectionLoop: { autoIterate: boolean; loopName: string } | undefined,
  collectionId: string | null,
  flow: FlowConfig,
  currentScreenInfo: { subcategoryRoute: string; subSubcategoryRoute: string },
  reviewMode: boolean
) => {
  if (collectionLoop && !collectionLoop.autoIterate && collectionId) {
    // If we're in a collection loop, we have to go the collection data view
    return makeLoopRoute(`/data-view/loop/${encodeURIComponent(collectionLoop.loopName)}`);
  } else {
    const subcategory = flow.subcategoriesByRoute.get(currentScreenInfo.subcategoryRoute);
    const subSubcategory = subcategory?.subSubcategories.find(
      ({ fullRoute }) => fullRoute === currentScreenInfo.subSubcategoryRoute
    );

    const dataViewRoute = `/data-view${currentScreenInfo.subcategoryRoute}`;
    return reviewMode ? makeRoutable(`${dataViewRoute}#${subSubcategory?.routeSuffix}`) : makeRoutable(dataViewRoute);
  }
};

export default getNextScreen;
