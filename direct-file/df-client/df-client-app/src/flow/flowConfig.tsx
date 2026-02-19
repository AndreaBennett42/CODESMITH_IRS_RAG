import {
  FlowChild,
  FlowNode,
  ScreenNode,
  isCategory,
  isSubcategory,
  isSubSubcategory,
  isCollectionLoop,
  isGate,
  isScreen,
  isAssertion,
} from './flowDeclarations.js';
import { assertNever } from 'assert-never';
import { Children, FC, ReactNode, createContext, useContext, useMemo } from 'react';
import { ConcretePath } from '@irs/js-factgraph-scala';
import { RawCondition } from './Condition.js';
import { Path } from './Path.js';
import { ScreenConfig } from './ScreenConfig.js';

import { IconName } from '../components/IconDisplay/IconDisplay.js';
import { checkScreenForRequiredButtons, checkScreenForSubSubCategory, checkScreenHeadings } from './flowHelpers.js';
import flowNodes from './flow.js';
import { AbsolutePath, WritableAbsolutePath } from '../fact-dictionary/Path.js';
import { DataItemConfig } from '../components/checklist/ChecklistSubcategory/DataReveal.js';
import { HeadingLevel } from '@trussworks/react-uswds';

export interface FlowCategory {
  route: string;
  subcategories: FlowSubcategory[];
}

/** We store a tree of screens and what gates they are behind */
export interface GatedTreeNode {
  gates: RawCondition[];
  screens: (ScreenConfig | GatedTreeNode)[];
}

export interface FlowSubcategory {
  route: string;
  categoryRoute: string;
  completionCondition: RawCondition | RawCondition[];
  screens: ScreenConfig[];
  // subSubCategories contains all subsubcategories that are not in loops.
  // subSubCategories in loops will appear in their respective loop.
  subSubcategories: FlowSubSubcategory[];
  loops: FlowCollectionLoop[];
  subSubcategoriesAndLoops: Array<FlowSubSubcategory | FlowCollectionLoop>;
  // if the entire subcategory refers to a single collection name
  // we place the collection name in the subcategory
  collectionName?: ConcretePath;
  hasDataView: boolean;
  isSignAndSubmit?: boolean;
  displayOnlyIf?: RawCondition | RawCondition[];
  assertions: FlowAssertion[];
  treeNodes: GatedTreeNode[];
  lockFutureSectionsIfCollectionItemsIncomplete?: boolean;
  dataItems?: DataItemConfig[];
}

export interface FlowAssertion {
  type: `success` | `warning` | `inactive` | `info`;
  subSubCategoryRoute?: string;
  subcategoryRoute: string;
  i18nKey: string;
  condition?: RawCondition;
  conditions?: RawCondition[];
  editRoute?: string;
}

export interface ItemAssertion {
  i18nKey: string;
  collectionItemI18nKey?: string;
  resultI18nKey?: string;
  outcomeI18nKey?: string;
  outcomeReviewRoute?: string;
  subSubCategoryToEdit?: string;
  type: `success` | `warning` | `inactive` | `info`;
  conditions?: RawCondition[];
}

export type SubSubcategoryBorderStyle = 'none' | 'normal' | 'heavy' | 'auto';
export interface FlowSubSubcategory {
  subcategoryRoute: string;
  fullRoute: string;
  routeSuffix: string;
  screens: ScreenConfig[];
  loopName?: string;
  editable?: boolean;
  hidden?: boolean;
  borderStyle?: SubSubcategoryBorderStyle;
  headingLevel?: HeadingLevel;
}

export interface FlowCollectionLoop {
  loopName: string;
  autoIterate: boolean;
  screens: ScreenConfig[];
  subSubcategories: FlowSubSubcategory[];
  subcategoryRoute: string;
  fullRoute: string;
  isInner: boolean; // indicates it is below subcat level
  iconName?: IconName;
  collectionName: ConcretePath; // TODO this should be a Path
  collectionItemCompletedCondition?: RawCondition;
  donePath?: WritableAbsolutePath; // This needs to be writable because we will update it when the user selects "done"
  knockoutRoute?: string;
  isImportedFactPath?: AbsolutePath;
  importedFlowStartRoute?: string;
  importedFlowDonePath?: AbsolutePath;
  importedRouteOverride?: string;
  hideCardLabel2Condition?: RawCondition;
  shouldSeeHubCompletionBtnsPath?: AbsolutePath;
  dataViewSections?: {
    // Cards on the hub are separated into sections
    resultI18nKey?: string;
    i18nKey: string;
    i18nModalKey?: string;
    condition?: RawCondition;
    conditions?: RawCondition[];
    itemAssertions?: ItemAssertion[];
  }[];
  dataReveals?: DataItemConfig[];
}

export interface FlowConfig {
  readonly screens: ScreenConfig[];
  readonly categories: FlowCategory[];

  readonly screensByRoute: Map<string, ScreenConfig>;
  readonly subcategoriesByRoute: Map<string, FlowSubcategory>;
  readonly subsubcategoriesByRoute: Map<string, FlowSubSubcategory>;
  readonly collectionLoopsByName: Map<string, FlowCollectionLoop>;
}

interface FlowContext {
  screenData: ScreenNode;
  categoryRoute: string;
  subcategoryRoute: string;
  subSubcategoryRoute: string;
  collectionContext?: ConcretePath;
  collectionLoop?: {
    loopName: string;
    autoIterate: boolean;
    isInner: boolean;
    fullRoute: string;
    dataViewOverride?: string;
  };
  conditions: RawCondition[];
  dataPreviewRoute: string;
}

interface PartialFlowContext extends Partial<FlowContext> {
  conditions: RawCondition[];
  currentTreeNode: GatedTreeNode;
}

function createScreenConfigFromAddScreenArgs(screenConfig: FlowContext) {
  const localCondition: RawCondition | undefined = screenConfig.screenData.props.condition;
  const allConditions = localCondition ? [...screenConfig.conditions, localCondition] : screenConfig.conditions;
  const routeAutomatically = !(screenConfig.screenData.props.routeAutomatically === false);
  const { actAsDataView, isKnockout, hideBreadcrumbs, hasScreenRouteOverride } = screenConfig.screenData.props;
  return new ScreenConfig(
    screenConfig.screenData.props.children,
    screenConfig.screenData.props.route,
    screenConfig.dataPreviewRoute,
    screenConfig.subSubcategoryRoute,
    screenConfig.subcategoryRoute,
    screenConfig.categoryRoute,
    allConditions,
    localCondition,
    screenConfig.collectionContext,
    screenConfig.collectionLoop,
    routeAutomatically,
    actAsDataView,
    screenConfig.screenData.props.alertAggregatorType,
    isKnockout,
    hideBreadcrumbs,
    hasScreenRouteOverride
  );
}

// N.B. I wish this was a function instead of a one-use class,
// but that will require a refactor of the building logic.
class FlowConfigBuilder {
  readonly screens: ScreenConfig[] = [];
  readonly categories: FlowCategory[] = [];
  readonly screensByRoute: Map<string, ScreenConfig> = new Map<string, ScreenConfig>();
  readonly subcategoriesByRoute: Map<string, FlowSubcategory> = new Map();
  readonly subsubcategoriesByRoute: Map<string, FlowSubSubcategory> = new Map();
  readonly collectionLoopsByName: Map<string, FlowCollectionLoop> = new Map();

  private addScreen(screenConfig: ScreenConfig, treeNode: GatedTreeNode, subsubCategoryRoute?: string): void {
    checkScreenForRequiredButtons(screenConfig);
    checkScreenHeadings(screenConfig);

    this.screens.push(screenConfig);
    if (this.screensByRoute.has(screenConfig.screenRoute)) {
      throw new Error(`Multiple screen configs for route ${screenConfig.screenRoute}`);
    }
    this.screensByRoute.set(screenConfig.screenRoute, screenConfig);

    // Add the screen to relevant subcategory and collection loops
    const subcategory = this.subcategoriesByRoute.get(screenConfig.subcategoryRoute);
    if (!subcategory) {
      throw new Error(
        `Screen ${screenConfig.screenRoute} has non-existent subcategory route ${screenConfig.subcategoryRoute}`
      );
    }

    subcategory.screens.push(screenConfig);
    if (screenConfig.collectionLoop) {
      const loop = this.collectionLoopsByName.get(screenConfig.collectionLoop.loopName);
      if (!loop) {
        throw new Error(
          `Screen ${screenConfig.screenRoute} has non-existent loop ${screenConfig.collectionLoop.loopName}`
        );
      }
      loop.screens.push(screenConfig);
    }
    const needsSubSubCategory = checkScreenForSubSubCategory(screenConfig);
    if (needsSubSubCategory && !subsubCategoryRoute) {
      throw new Error(`${screenConfig.screenRoute} needs a subsubcategory`);
    }
    if (subsubCategoryRoute) {
      const subsubCategory = this.subsubcategoriesByRoute.get(subsubCategoryRoute);
      if (!subsubCategory) {
        throw new Error(`${screenConfig.screenRoute} had non-existent subsubcategory ${subsubCategoryRoute}`);
      }
      subsubCategory.screens.push(screenConfig);
    }

    // Last, we place the screen into the tree data structure that contains the subcategory's
    // gate structure
    if (screenConfig.localCondition) {
      treeNode.screens.push({
        gates: [screenConfig.localCondition],
        screens: [screenConfig],
      });
    } else {
      treeNode.screens.push(screenConfig);
    }
  }

  private addSubSubcategory({
    subcategoryRoute,
    fullRoute,
    loopName,
    routeSuffix,
    editable,
    hidden,
    borderStyle,
    headingLevel,
  }: Omit<FlowSubSubcategory, 'screens'>) {
    const subcategory = this.subcategoriesByRoute.get(subcategoryRoute);
    if (!subcategory) {
      throw new Error(`Subsubcategory ${fullRoute} had no subcategory ${subcategoryRoute}`);
    }
    const subsubCategory = {
      subcategoryRoute,
      fullRoute,
      routeSuffix,
      screens: [],
      loopName,
      editable,
      hidden,
      borderStyle,
      headingLevel,
    };
    if (this.subsubcategoriesByRoute.has(fullRoute)) {
      // Do not add a new SubSubCategory if it already exists
      return;
    } else {
      this.subsubcategoriesByRoute.set(fullRoute, subsubCategory);
    }
    if (loopName) {
      const loop = this.collectionLoopsByName.get(loopName);
      if (!loop) {
        throw new Error(`Subsubcategory ${fullRoute} had no loop ${loopName}`);
      }
      loop.subSubcategories.push(subsubCategory);
    } else {
      subcategory.subSubcategories.push(subsubCategory);
      subcategory.subSubcategoriesAndLoops.push(subsubCategory);
    }
  }

  private addSubcategory(
    args: Omit<FlowSubcategory, 'screens' | 'subSubcategories' | 'loops' | 'subSubcategoriesAndLoops'>
  ) {
    const subcategory: FlowSubcategory = {
      ...args,
      screens: [],
      subSubcategories: [],
      loops: [],
      subSubcategoriesAndLoops: [],
    };
    const category = this.categories.find((c) => c.route === args.categoryRoute);
    if (!category) {
      throw new Error(`Subcategory ${args.route} had no category ${args.categoryRoute}`);
    }
    category.subcategories.push(subcategory);
    this.subcategoriesByRoute.set(args.route, subcategory);
  }

  private addAssertion(args: FlowAssertion) {
    const assertion = { ...args };
    const subcategory = this.subcategoriesByRoute.get(args.subcategoryRoute);
    if (!subcategory) {
      throw new Error(`Assertion ${args} had no subcategory ${args.subcategoryRoute}`);
    }
    subcategory.assertions.push(assertion);
  }

  private addCollectionLoop(args: Omit<FlowCollectionLoop, 'screens' | 'subSubcategories'>) {
    if (this.collectionLoopsByName.has(args.loopName)) {
      throw new Error(`Multiple loop configs for loop name ${args.loopName}`);
    }
    const loop = { ...args, screens: [], subSubcategories: [] };
    this.collectionLoopsByName.set(args.loopName, loop);
    const subcategory = this.subcategoriesByRoute.get(args.subcategoryRoute);
    if (!subcategory) {
      throw new Error(`Loop ${args.loopName} had no subcategory ${args.subcategoryRoute}`);
    }
    subcategory.loops.push(loop);
    subcategory.subSubcategoriesAndLoops.push(loop);
  }

  private addCategory(args: Omit<FlowCategory, 'subcategories'>) {
    this.categories.push({ ...args, subcategories: [] });
  }

  public parseFlow(flow: FlowNode) {
    this.parseFlowRecursive(flow.props.children, {
      conditions: [],
      collectionContext: undefined,
      collectionLoop: undefined,
      currentTreeNode: {
        screens: [],
        gates: [],
      },
    });
  }

  private parseFlowRecursive(currentTree: FlowChild | FlowChild[], context: PartialFlowContext): void {
    if (Array.isArray(currentTree)) {
      currentTree.forEach((node) => {
        this.parseFlowRecursive(node, context);
      });
    } else if (isScreen(currentTree)) {
      const screenConfig = {
        ...context,
        screenData: currentTree,
      } as FlowContext;
      // n.b. This feels so wrong, to have a :void recursive function, but like
      // it also feels silly to return configs just to loop over them
      this.addScreen(
        createScreenConfigFromAddScreenArgs(screenConfig),
        context.currentTreeNode,
        context.subSubcategoryRoute
      );
    } else if (isCategory(currentTree)) {
      const categoryRoute = `/flow/${currentTree.props.route}`;
      const initialContext: PartialFlowContext = {
        ...context,
        categoryRoute,
      };

      this.addCategory({ route: categoryRoute });
      Children.forEach(currentTree.props.children, (node) => this.parseFlowRecursive(node, initialContext));
    } else if (isSubcategory(currentTree)) {
      const categoryRoute = context[`categoryRoute`];
      if (!categoryRoute) {
        throw new Error(`Subcategory ${currentTree.props.route} had no category route`);
      }
      const subcategoryRoute = `${context[`categoryRoute`]}/${currentTree.props.route}`;
      const subcategoryCompletionCondition = currentTree.props.completeIf;
      const subcategoryCollectionContext = currentTree.props.collectionContext;
      const subcategoryGates = currentTree.props.displayOnlyIf
        ? Array.isArray(currentTree.props.displayOnlyIf)
          ? currentTree.props.displayOnlyIf
          : [currentTree.props.displayOnlyIf]
        : [];
      const currentTreeNode = {
        screens: [],
        gates: subcategoryGates,
      };
      const updatedContext: PartialFlowContext = {
        ...context,
        currentTreeNode,
        subcategoryRoute,
        ...(subcategoryCollectionContext && {
          collectionContext: Path.concretePath(subcategoryCollectionContext, null),
        }),
      };
      this.addSubcategory({
        route: subcategoryRoute,
        completionCondition: subcategoryCompletionCondition,
        displayOnlyIf: currentTree.props.displayOnlyIf,
        categoryRoute,
        hasDataView: !currentTree.props.skipDataView,
        // eslint-disable-next-line eqeqeq
        isSignAndSubmit: currentTree.props.isSignAndSubmit == true,
        lockFutureSectionsIfCollectionItemsIncomplete: currentTree.props.lockFutureSectionsIfCollectionItemsIncomplete,
        collectionName: subcategoryCollectionContext
          ? Path.concretePath(subcategoryCollectionContext, null)
          : undefined,
        assertions: [],
        treeNodes: [currentTreeNode],
        dataItems: currentTree.props.dataItems,
      });
      Children.forEach(currentTree.props.children, (node) => this.parseFlowRecursive(node, updatedContext));
    } else if (isAssertion(currentTree)) {
      const subcategoryRoute = context.subcategoryRoute;
      const subSubCategoryRoute = context.subSubcategoryRoute;
      if (!subcategoryRoute) {
        throw new Error(`Assertion ${currentTree} is not in a subcategory`);
      }
      const assertion = {
        type: currentTree.props.type,
        subcategoryRoute: subcategoryRoute,
        subSubCategoryRoute: subSubCategoryRoute,
        i18nKey: currentTree.props.i18nKey,
        condition: currentTree.props.condition,
        conditions: currentTree.props.conditions,
        editRoute: currentTree.props.editRoute,
        children: [],
      };

      this.addAssertion(assertion);
    } else if (isCollectionLoop(currentTree)) {
      const subcategoryRoute = context.subcategoryRoute;
      if (!subcategoryRoute) {
        throw new Error(`Loop ${currentTree.props.loopName} is not in a subcategory`);
      }
      const collectionContext = currentTree.props.collection
        ? Path.concretePath(currentTree.props.collection, null)
        : context.collectionContext;
      if (!collectionContext) {
        throw new Error(`Loop ${currentTree.props.loopName} has no collection context`);
      }

      // If inner loop check location
      const isInner = currentTree.props.isInner === true;
      let fullRoute: string;
      if (isInner) {
        if (!context.subSubcategoryRoute) {
          throw new Error(`Inner loops should be in a subsubcategory`);
        }
        fullRoute = context.subSubcategoryRoute;
      } else {
        fullRoute = subcategoryRoute;
      }

      const updatedContext: PartialFlowContext = {
        ...context,
        collectionLoop: {
          loopName: currentTree.props.loopName,
          autoIterate: currentTree.props.autoIterate === true,
          isInner,
          fullRoute,
        },
        collectionContext,
      };
      this.addCollectionLoop({
        loopName: currentTree.props.loopName,
        autoIterate: currentTree.props.autoIterate === true,
        subcategoryRoute,
        fullRoute,
        iconName: currentTree.props.iconName,
        collectionName: collectionContext,
        donePath: currentTree.props.donePath,
        isInner: currentTree.props.isInner === true,
        dataViewSections: currentTree.props.dataViewSections,
        dataReveals: currentTree.props.dataReveals,
        collectionItemCompletedCondition: currentTree.props.collectionItemCompletedCondition,
        hideCardLabel2Condition: currentTree.props.hideCardLabel2Condition,
        shouldSeeHubCompletionBtnsPath: currentTree.props.shouldSeeHubCompletionBtnsPath,
        knockoutRoute: currentTree.props.knockoutRoute,
        isImportedFactPath: currentTree.props.isImportedFactPath,
        importedFlowStartRoute: currentTree.props.importedFlowStartRoute,
        importedFlowDonePath: currentTree.props.importedFlowDonePath,
        importedRouteOverride: currentTree.props.importedRouteOverride,
      });
      Children.forEach(currentTree.props.children, (node) => this.parseFlowRecursive(node, updatedContext));
    } else if (isGate(currentTree)) {
      const conditions = [...context.conditions, currentTree.props.condition];
      const newTreeNode: GatedTreeNode = {
        screens: [],
        gates: [currentTree.props.condition],
      };
      context.currentTreeNode.screens.push(newTreeNode);
      const updatedContext: PartialFlowContext = {
        ...context,
        conditions,
        currentTreeNode: newTreeNode,
      };
      Children.forEach(currentTree.props.children, (node) => this.parseFlowRecursive(node, updatedContext));
    } else if (isSubSubcategory(currentTree)) {
      const fullRoute = `${context.subcategoryRoute}/${currentTree.props.route}`;
      const subSubcategoryCollectionContext = currentTree.props.collectionContext;

      const updatedContext: PartialFlowContext = {
        ...context,
        subSubcategoryRoute: fullRoute,
        ...(subSubcategoryCollectionContext && {
          collectionContext: Path.concretePath(subSubcategoryCollectionContext, null),
        }),
      };
      if (!context.subcategoryRoute) {
        throw new Error(
          `Subsubcategory ${currentTree.props.route} had no subcategory route ${context.subSubcategoryRoute}`
        );
      }
      this.addSubSubcategory({
        subcategoryRoute: context.subcategoryRoute,
        fullRoute,
        loopName: context.collectionLoop?.loopName,
        routeSuffix: currentTree.props.route,
        editable: currentTree.props.editable !== false,
        hidden: currentTree.props.hidden === true,
        borderStyle: currentTree.props.borderStyle,
        headingLevel: currentTree.props.headingLevel,
      });
      Children.forEach(currentTree.props.children, (node) => this.parseFlowRecursive(node, updatedContext));
    } else {
      assertNever(currentTree);
    }
  }
}

export function createFlowConfig(flowData: FlowNode): FlowConfig {
  const builder = new FlowConfigBuilder();
  builder.parseFlow(flowData);
  return {
    categories: builder.categories,
    subcategoriesByRoute: builder.subcategoriesByRoute,
    screens: builder.screens,
    screensByRoute: builder.screensByRoute,
    collectionLoopsByName: builder.collectionLoopsByName,
    subsubcategoriesByRoute: builder.subsubcategoriesByRoute,
  };
}

const FlowContext = createContext<FlowConfig>({
  ...createFlowConfig(flowNodes),
});

export const FlowContextProvider: FC<{
  flowData?: FlowNode;
  children: ReactNode[] | ReactNode;
}> = (props) => {
  const { flowData } = props;
  const flowConfig = useMemo(() => {
    return createFlowConfig(flowData || flowNodes);
  }, [flowData]);
  return <FlowContext.Provider value={flowConfig}>{props.children}</FlowContext.Provider>;
};

export const useFlow = () => {
  return useContext(FlowContext);
};
