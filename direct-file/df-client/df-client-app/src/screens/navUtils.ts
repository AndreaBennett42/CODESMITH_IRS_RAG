import { FactGraph, ConcretePath, scalaListToJsArray } from '@irs/js-factgraph-scala';
import { AbsolutePath } from '../fact-dictionary/Path.js';
import { Path } from '../flow/Path.js';

export function getUrlSearchParams(
  currentParams: URLSearchParams | null,
  collectionContext: string | undefined,
  collectionId: string | null,
  reviewMode?: boolean
) {
  const newParams = new URLSearchParams({
    ...(currentParams && Object.fromEntries(currentParams)),
    ...(collectionContext &&
      collectionId &&
      Object.fromEntries(new URLSearchParams([[collectionContext, collectionId || ``]]))),
  });
  // Doing this outside of the construction lets us fully delete reviewMode if it's set to false instead of leaving it
  // set to false in the url parameters.
  if (reviewMode !== undefined) {
    if (reviewMode) {
      newParams.set(`reviewMode`, `true`);
    } else {
      newParams.delete(`reviewMode`);
    }
  }
  return newParams.toString();
}

// This function gets the first collection item id that doesn't satisfy the fact path.
// This is used to route the TP to additional screens from the CollectionItemDataView.
// This logic could presumably go inside `getNextScreen`, but that file is very big already and this logic
// is self-contained to the flow of:
//   - TP selects to import data from a data preview screen ->
//   - TP is routed to screens within the collection that require manual review (outside of flow) ->
//   - TP sees data view ->
//   - TP sees breather outro screen (resume flow)
export const getCollectionItemThatNeedsManualReview = (
  factGraph: FactGraph,
  collectionContext: string,
  factPath: AbsolutePath
) => {
  const collectionResult = factGraph.get(collectionContext as ConcretePath);
  const collectionItems: string[] = collectionResult.complete
    ? scalaListToJsArray(collectionResult.get.getItemsAsStrings())
    : [];

  const foundItem =
    collectionItems.find((id) => {
      const completedFlowFactPath = Path.concretePath(factPath, id);
      const hasCompletedFlow = factGraph.get(completedFlowFactPath).hasValue
        ? factGraph.get(completedFlowFactPath).get
        : false;
      return hasCompletedFlow === false ? id : null;
    }) || null;
  return foundItem;
};

export const routeToScreenWithItemId = (route: string, collectionContext: string, id: string) => {
  return `${route}?&${encodeURIComponent(collectionContext as string)}=${id}`;
};
