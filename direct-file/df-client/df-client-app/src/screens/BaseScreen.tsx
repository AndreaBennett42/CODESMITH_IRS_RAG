import { useParams, useNavigate, useSearchParams, Navigate } from 'react-router-dom';
import Screen from '../components/Screen.js';
import { Path } from '../flow/Path.js';
import { Path as FDPath } from '../fact-dictionary/Path.js';
import getNextScreen from './getNextScreen.js';
import { useFlow } from '../flow/flowConfig.js';
import { useFactGraph } from '../factgraph/FactGraphContext.js';
import { useCallback } from 'react';
import { ConcretePath, FactGraph } from '@irs/js-factgraph-scala';
import { ScreenConfig } from '../flow/ScreenConfig.js';

/**
 * The component renders a screen within a flow of screens based on the current
 * URL. The flow of screens is defined in the flow module, and the screen
 * configurations are retrieved using the getAllScreensByRoute function
 * defined in the flowUtils module.
 *
 * The component uses the useParams and useNavigate hooks provided by the
 * react-router-dom library to get the current URL and to navigate to the next
 * or previous screen in the flow.
 *
 * The getNextScreen function is defined in the getNextScreen module and
 * determines the next screen in the flow based on the current screen and
 * the user's input.
 *
 * Finally, the BaseScreen component returns the React component corresponding
 * to the current URL and includes a "Previous" button that allows the user to
 * navigate to the previous screen in the flow.
 *
 * @returns {JSX}
 */

function BaseScreen() {
  const { factGraph } = useFactGraph();
  const flow = useFlow();
  const params = useParams();
  const [searchParams] = useSearchParams();
  const currentUrl = `/flow/${params[`*`]}`;
  const navigate = useNavigate();
  const screenInfo = flow.screensByRoute.get(currentUrl) as ScreenConfig;
  const categoryRoute = currentUrl.substring(0, currentUrl.lastIndexOf(`/`));
  const subcategory = flow.subcategoriesByRoute.get(categoryRoute);
  const collectionId = getCollectionId(factGraph, searchParams, screenInfo?.collectionContext);
  const isReviewMode = searchParams.get(`reviewMode`);

  const gotoNextScreen = useCallback(() => {
    if (screenInfo) {
      const nextPath = getNextScreen(screenInfo, factGraph, collectionId, flow, {
        // eslint-disable-next-line eqeqeq
        navigateToDataViewAtEndOfSubSubCategory: isReviewMode == `true`,
      });
      console.log(`🚀 ~ gotoNextScreen ~ nextPath:`, nextPath);
      navigate(nextPath.routable.fullRoute(nextPath.collectionId), {
        state: {
          from: currentUrl,
        },
      });
    }
  }, [screenInfo, factGraph, collectionId, flow, isReviewMode, navigate, currentUrl]);

  if (!screenInfo || !isValidCollectionInURL(factGraph, searchParams, screenInfo?.collectionContext)) {
    // if we're at an invalid URL, navigate to a 404 page
    return <Navigate to='../not-found' replace />;
  }

  if (!subcategory) throw new Error(`No subcategory found`);

  return (
    <Screen
      collectionContext={screenInfo.collectionContext}
      screenRoute={screenInfo.screenRoute}
      screenContent={screenInfo.content}
      collectionId={collectionId}
      gotoNextScreen={gotoNextScreen}
      setFactActionPaths={screenInfo.setActions}
      key={screenInfo.fullRoute(collectionId)}
      alertAggregatorType={screenInfo.alertAggregatorType}
    />
  );
}
export default BaseScreen;

/**
 * Gets the collectionId associated with current screen by searching the URL
 * params for the collectionName provided
 */
export function getCollectionId(
  factGraph: FactGraph,
  searchParams: URLSearchParams,
  collectionName: ConcretePath | undefined
): string | null {
  // eslint-disable-next-line eqeqeq
  if (collectionName == undefined) {
    return null;
  } else if (collectionName === `/primaryFiler` || collectionName === `/secondaryFiler`) {
    const filerResult = factGraph.get(collectionName);
    if (filerResult.complete) {
      return filerResult.get.idString;
    } else {
      return null;
    }
  } else {
    return searchParams.get(collectionName);
  }
}

export function isValidCollectionInURL(
  factGraph: FactGraph,
  searchParams: URLSearchParams,
  collectionName: ConcretePath | undefined,
  collectionItemId?: string | null
) {
  if (collectionName !== undefined) {
    const itemId = collectionItemId || searchParams.get(collectionName as string);
    let fgCollectionName = collectionName as string;
    if (collectionName === `/primaryFiler` || collectionName === `/secondaryFiler`) {
      fgCollectionName = `/filers`;
    }
    if (itemId !== null) {
      try {
        // This returns a result with collectionItem IF the item exists
        const fgPath = `${fgCollectionName}/*` as FDPath;
        factGraph.get(Path.concretePath(fgPath, itemId));
      } catch (error) {
        if (error?.toString().includes(`not found`)) {
          return false;
        }
      }
    }
  }
  return true;
}
