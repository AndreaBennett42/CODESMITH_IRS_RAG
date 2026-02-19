import { FactGraph, ScalaList, convertCollectionToArray, scalaListToJsArray } from '@irs/js-factgraph-scala';
import { FlowCollectionLoop, FlowSubcategory, FlowSubSubcategory } from './flowConfig.js';
import { FactConfig, ScreenContentConfig, contentConfigIsFactConfig } from './ContentDeclarations.js';
import { Path } from './Path.js';
import { ScreenConfig, SetFactActionConfig } from './ScreenConfig.js';
import { Path as FDPath } from '../fact-dictionary/Path.js';
import { conditionsPass } from '../utils/condition.js';
import { Condition } from './Condition.js';
import { getInnerLoopHub } from '../misc/collectionLoopHelpers.js';

export interface ScreenConfigWithCollectionId {
  screen: ScreenConfig | undefined;
  id: null | string;
}

export function findFirstIncompleteScreenOfSubcategory(
  subcategory: FlowSubcategory,
  factGraph: FactGraph,
  collectionId: string | null
): ScreenConfigWithCollectionId {
  for (const sscOrLoop of subcategory.subSubcategoriesAndLoops) {
    const isLoop = sscOrLoop.loopName !== undefined;
    if (isLoop) {
      const loop = sscOrLoop as FlowCollectionLoop;
      if (hasAtLeastOneIncompleteCollectionItem(loop, factGraph)) {
        const result = factGraph.get(loop.collectionName);
        const collectionItemIds = convertCollectionToArray(result.get as ScalaList<string>);
        for (const itemId of collectionItemIds) {
          const foundScreen = findFirstIncompleteScreenOfLoop(loop, factGraph, itemId);
          if (foundScreen) {
            return { screen: foundScreen, id: itemId };
          }
        }
      }

      // If we didn't find any incomplete screens in the loops, but we found a loop with an
      // incomplete done condition we need to find the collection hub screen for that loop.
      // This usually happens when the user has not completed the collection item manager.
      if (loop.donePath) {
        const collectionDone =
          factGraph.get(Path.concretePath(loop.donePath, null)).complete &&
          // eslint-disable-next-line eqeqeq
          factGraph.get(Path.concretePath(loop.donePath, null)).get == true;
        if (!collectionDone) {
          const subCatWithHub = subcategory.subSubcategories.find((ssc) => {
            return ssc.fullRoute === loop.fullRoute;
          });
          //  Get the hub screen for the loop from the subcategory.
          if (subCatWithHub) {
            const screen = getInnerLoopHub(subCatWithHub);
            if (screen?.isAvailable(factGraph, collectionId)) {
              return { screen, id: collectionId };
            }
          }
        }
      }
    } else {
      const ssc = sscOrLoop as FlowSubSubcategory;
      const filteredScreens = ssc.screens.filter((screen) => screen.isAvailable(factGraph, collectionId));
      const foundScreen = findFirstIncompleteScreen(filteredScreens, factGraph, collectionId);
      if (foundScreen) {
        return { screen: foundScreen, id: collectionId };
      }
    }
  }

  return { screen: undefined, id: null };
}

export function findFirstIncompleteScreenOfLoop(
  loop: FlowCollectionLoop,
  factGraph: FactGraph,
  collectionId: string | null
) {
  const { collectionItemCompletedCondition } = loop;
  // Loop does not define completion && we need to find the first incomplete screen
  // or loop defines completion and the collection item is not complete
  // This check primarily helps us find the correct next screen for inner collection loops
  // where we iterate over the collection items. Without this we might return the first optional screen
  // skipped by the user in the first collection item even if the user has completed the collection item.
  if (
    !collectionItemCompletedCondition ||
    !new Condition(collectionItemCompletedCondition).evaluate(factGraph, collectionId)
  ) {
    return findFirstIncompleteScreen(
      loop.screens.filter((screen) => screen.isAvailable(factGraph, collectionId)),
      factGraph,
      collectionId
    );
  } else return undefined;
}

export function hasAtLeastOneIncompleteCollectionItem(loop: FlowCollectionLoop, factGraph: FactGraph) {
  // If the subsection has a loop which defines the completion of its items,
  // and any item in the collection is incomplete, return false.
  const { collectionItemCompletedCondition, collectionName } = loop;
  if (!collectionItemCompletedCondition) {
    return false;
  }

  const collectionItemsResult = factGraph.get(collectionName);
  const collectionItems: string[] = collectionItemsResult.complete
    ? scalaListToJsArray(collectionItemsResult.get.getItemsAsStrings())
    : [];
  return collectionItems.some((itemId) => {
    const itemIsComplete = new Condition(collectionItemCompletedCondition).evaluate(factGraph, itemId);
    return !itemIsComplete;
  });
}

function findFirstIncompleteScreen(screens: ScreenConfig[], factGraph: FactGraph, collectionId: string | null) {
  return screens.find(
    (screen) =>
      screen.content
        .filter((c): c is FactConfig => contentIsAvailableRequiredWritableFactConfig(c, factGraph, collectionId))
        .some((fact) => !factGraph.get(Path.concretePath(fact.props.path, collectionId)).complete) ||
      // It's possible we can end up in a situation where the checklist says the subcategory is incomplete
      // due to an incomplete collection item's condition that depends on a fact action
      // (seeing the last available screen, for example).
      // Therefore, we need to check "required" fact actions.
      screen.setActions
        .filter((c) => contentIsAvailableRequiredFactAction(c, factGraph, collectionId))
        .some((fact) => !factGraph.get(Path.concretePath(fact.path, collectionId)).complete)
  );
}

export function subcategoryHasSomeCompletedFacts(
  subcategory: FlowSubcategory,
  factGraph: FactGraph,
  collectionId: string | null
) {
  return (
    subcategory.screens.some(
      (screen) =>
        screen.isAvailable(factGraph, collectionId) &&
        // eslint-disable-next-line eqeqeq
        screen.collectionLoop == undefined &&
        screen.content
          .filter((c): c is FactConfig => contentIsAvailableRequiredWritableFactConfig(c, factGraph, collectionId))
          .some((fact) => factGraph.get(Path.concretePath(fact.props.path, collectionId)).complete)
      // We exclude autoiterating loops because they can have "completed" collections even though
      // the user has not entered the flow for the subcategory. (Ex: Dependents' eligibility determined
      // in the family and household section can result in completed states for /deceasedEitcEligibleQcCollection
      // and /unclaimedEITCQcsCollection.)
    ) || subcategory.loops.some((loop) => !loop.autoIterate && collectionLoopHasAtLeastOneMember(loop, factGraph))
  );
}

// TODO: https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/3122
// We should have a better way of skipping over fields that the user has declined to answer.
// the user can still fill out these fields by modifying the basic info on a data view -- we just won't bring them
// to the middle initial field when they hit "continue"
export const SKIPPED_INCOMPLETE_FACTS: FDPath[] = [
  `/familyAndHousehold/*/writableMiddleInitial`,
  `/filers/*/writableMiddleInitial`,
  `/filers/*/writableSuffix`,
  `/familyAndHousehold/*/writableSuffix`,
  // TODO: Remove this - this is a temporary hack to allow us to roll out presetting the primary filer tin on the
  // backend. The long term fix is to make this non editable on the fact graph, but I don't want to make that
  // change until we've verified TIN comes through in every environment.
  // https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/10162?show=15482
  `/primaryFiler/tin`,
  // TODO: Remove this - this is a temporary solution to importing W2s to not show incomplete errors for
  // data which we won't have.
  `/formW2s/*/writableHasBox14Codes`,
  `/formW2s/*/writableState`,
  `/formW2s/*/writableStateEmployerId`,
  `/formW2s/*/writableStateWages`,
  `/formW2s/*/writableStateWithholding`,
  `/formW2s/*/writableLocalWages`,
  `/formW2s/*/writableLocalWithholding`,
  `/formW2s/*/writableLocality`,
  `/form1099Rs/*/writablePayerNameLine2`,
];

export function contentIsAvailableRequiredWritableFactConfig(
  c: ScreenContentConfig,
  factGraph: FactGraph,
  collectionId: string | null
): c is FactConfig {
  return (
    contentConfigIsFactConfig(c) &&
    conditionsPass(c.props, factGraph, collectionId) &&
    // Disabled, readonly, and data view-only props are all read only.
    // eslint-disable-next-line eqeqeq
    c.props.displayOnlyOn != `data-view` &&
    // eslint-disable-next-line eqeqeq
    c.props.readOnly != true &&
    // A FactSelect is always optional
    // eslint-disable-next-line eqeqeq
    c.componentName != `FactSelect` &&
    !SKIPPED_INCOMPLETE_FACTS.includes(c.props.path)
  );
}

export function contentIsAvailableRequiredFactAction(
  c: SetFactActionConfig,
  factGraph: FactGraph,
  collectionId: string | null
): c is SetFactActionConfig {
  return conditionsPass(c, factGraph, collectionId);
}

function collectionLoopHasAtLeastOneMember(loop: FlowCollectionLoop, factGraph: FactGraph) {
  const result = factGraph.get(loop.collectionName);
  const collectionItems: string[] = result.complete ? convertCollectionToArray(result.get as ScalaList<string>) : [];
  return collectionItems.length > 0;
}

export function checkScreenForRequiredButtons(screenConfig: ScreenConfig) {
  const requiredScreenButtons: Set<string> = new Set([
    `SaveAndOrContinueButton`,
    `SaveAndOrContinueAndSetFactButton`,
    `ExitButton`,
    `SubmitButton`,
    `CollectionItemManager`,
    `CollectionDataPreview`,
    `KnockoutButton`,
    `InternalLink`,
    `CollectionDataViewInternalLink`,
  ]);

  const screenHasRequiredScreenButton = screenConfig.content.some((obj) =>
    requiredScreenButtons.has(obj.componentName)
  );

  if (!screenHasRequiredScreenButton) {
    throw new Error(
      `Screen at route ${screenConfig.screenRoute} must have a one of these buttons: ${Array.from(
        requiredScreenButtons
      ).join(`, `)} `
    );
  }
}

export function checkScreenHeadings(screenConfig: ScreenConfig) {
  const headingComponents = screenConfig.content.filter((obj) => obj.componentName === `Heading`);

  if (
    headingComponents.length > 1 &&
    headingComponents.some((obj) => obj.props?.condition === undefined && obj.props?.conditions === undefined)
  ) {
    throw new Error(
      `Screen at route ${screenConfig.screenRoute} has more than one Heading but is missing conditions on each Heading.`
    );
  }

  if (headingComponents.length === 1 && headingComponents[0].props?.conditions) {
    throw new Error(
      // eslint-disable-next-line max-len
      `Screen at route ${screenConfig.screenRoute} has only one Heading with a condition. Maybe you need to either remove the condition or add another Heading :) ?`
    );
  }

  if (headingComponents.length === 0) {
    throw new Error(`Screen at route ${screenConfig.screenRoute} requires a Heading`);
  }
}

export function checkScreenForSubSubCategory(screenConfig: ScreenConfig) {
  return screenConfig.content.find((c): c is FactConfig => contentConfigIsFactConfig(c));
}
