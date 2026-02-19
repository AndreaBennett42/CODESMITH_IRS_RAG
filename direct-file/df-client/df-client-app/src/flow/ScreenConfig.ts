import { ConcretePath, FactGraph, convertCollectionToArray } from '@irs/js-factgraph-scala';
import { Children } from 'react';
import { RawCondition, Condition } from './Condition.js';
import {
  AllComponentTypes,
  FactActionSource,
  FlowComponentName,
  ScreenContentConfig,
  ScreenContentNode,
  contentConfigIsFactConfig,
  isFactAction,
} from './ContentDeclarations.js';
import { Path as FGPath } from '../fact-dictionary/Path.js';
import { getUrlSearchParams } from '../screens/navUtils.js';
import { AlertAggregatorType } from './flowDeclarations.js';
import { BATCH_NAME } from './batches.js';

export interface SetFactActionConfig {
  readonly path: FGPath;
  readonly source: FactActionSource;
  readonly condition?: RawCondition;
  readonly conditions?: RawCondition[];
}
export function buildScreenContentFromConfigNode(configNode: ScreenContentNode): ScreenContentConfig | undefined {
  if (Array.isArray(configNode)) {
    throw new Error(`buildFromConfigNode only operates on nodes, not lists of nodes`);
  }

  const componentName = configNode.type.name satisfies FlowComponentName;

  if (componentName === `SetFactAction`) {
    // TODO: Explain why we filter out `SetFactAction` here
    return undefined;
  }

  if (componentName === undefined || !Object.keys(AllComponentTypes).includes(componentName)) {
    throw new Error(`Expected ScreenContentNode, found ${componentName} instead`);
  }

  if (`children` in configNode.props && configNode.props.children) {
    const { children, ...props } = configNode.props;
    const childConfigs = (Array.isArray(children) ? children : [children]) as ScreenContentNode[];

    return {
      componentName,
      props: {
        ...props,
        childConfigs: childConfigs.map(buildScreenContentFromConfigNode),
      },
    } as ScreenContentConfig;
  }

  return {
    componentName,
    props: configNode.props,
  } as ScreenContentConfig;
}

export function buildFactActionFromConfigNode(configNode: ScreenContentNode): SetFactActionConfig | undefined {
  if (Array.isArray(configNode)) {
    throw new Error(`buildFromConfigNode only operates on nodes, not lists of nodes`);
  }
  if (isFactAction(configNode)) {
    const path = configNode.props.path;
    const source = configNode.props.source;
    const condition = configNode.props.condition;
    const conditions = configNode.props.conditions;
    return {
      path,
      source,
      condition,
      conditions,
    };
  }
  return undefined;
}

export interface RouteOptions {
  reviewMode?: boolean;
}

export class ScreenConfig {
  readonly route: string;
  readonly dataPreviewRoute: string;
  readonly subSubcategoryRoute: string;
  readonly subcategoryRoute: string;
  readonly categoryRoute: string;
  readonly screenRoute: string;
  readonly conditions: RawCondition[];
  readonly collectionContext: ConcretePath | undefined;
  readonly collectionLoop?: {
    loopName: string;
    autoIterate: boolean;
    isInner: boolean;
    fullRoute: string;
  };
  readonly batches: BATCH_NAME[];
  readonly routeAutomatically: boolean = false;
  readonly actAsDataView: boolean = false;
  readonly alertAggregatorType?: AlertAggregatorType = `screen`;
  readonly isKnockout: boolean = false;
  readonly hideBreadcrumbs: boolean = false;
  readonly hasScreenRouteOverride: boolean = false;
  private readonly _content: ScreenContentConfig[];
  private readonly _factPaths: FGPath[];
  private readonly _setActions: SetFactActionConfig[];
  constructor(
    configNode: ScreenContentNode[],
    route: string,
    dataPreviewRoute: string,
    subSubcategoryRoute: string,
    subcategoryRoute: string,
    categoryRoute: string,
    conditions: RawCondition[],
    public readonly localCondition?: RawCondition,
    collectionContext: ConcretePath | undefined = undefined,
    collectionLoop:
      | {
          loopName: string;
          autoIterate: boolean;
          isInner: boolean;
          fullRoute: string;
        }
      | undefined = undefined,
    routeAutomatically = false,
    actAsDataView = false,
    alertAggregatorType: AlertAggregatorType = `screen`,
    isKnockout = false,
    hideBreadcrumbs = false,
    hasScreenRouteOverride = false
  ) {
    this.route = route;
    this.dataPreviewRoute = dataPreviewRoute;
    this.subSubcategoryRoute = subSubcategoryRoute;
    this.subcategoryRoute = subcategoryRoute;
    this.categoryRoute = categoryRoute;
    this.collectionContext = collectionContext;
    this.conditions = conditions;
    this.screenRoute = `${this.subcategoryRoute}/${this.route}`;
    this.collectionLoop = collectionLoop;
    this.routeAutomatically = routeAutomatically;
    this.actAsDataView = actAsDataView;
    this.alertAggregatorType = alertAggregatorType;
    this.isKnockout = isKnockout;
    this.hideBreadcrumbs = hideBreadcrumbs;
    this.hasScreenRouteOverride = hasScreenRouteOverride;
    this._content = Children.map(configNode, buildScreenContentFromConfigNode).filter((n) => n !== undefined);
    this._setActions = Children.map(configNode, buildFactActionFromConfigNode).filter((n) => n !== undefined);

    const hasHeader = this._content.some((c) => c.componentName === `Heading`);
    if (!hasHeader) {
      // TODO: this would be nice as a lint rule to have in static analysis instead of being a runtime error
      // but at least it's a runtime error when the app starts up.
      throw new Error(`Screen at ${route} has no Heading configs`);
    }

    this._factPaths = this._content.filter(contentConfigIsFactConfig).map((content) => content.props.path);
    // Array.from // new Set uniques the batch names from the content
    this.batches = Array.from(new Set(this._content.flatMap((c) => c.props.batches || [])));
  }

  /**
   * Returns the full route for the screen, including any collectionId or reviewMode parameters
   * @param collectionId - The collectionId to use for the route
   * @param opts - Optional route options such as reviewMode
   * @returns The full route for the screen
   */
  fullRoute(collectionId: string | null, opts?: RouteOptions) {
    const currentParams = new URLSearchParams(window.location.search);
    const urlSearchParams = getUrlSearchParams(currentParams, this.collectionContext, collectionId, opts?.reviewMode);
    let route = `${this.screenRoute}${urlSearchParams && `?`}${urlSearchParams}`;
    if (this.hasScreenRouteOverride) {
      route = `/data-view${this.subSubcategoryRoute}`;
    }
    return route;
  }

  get content() {
    return [...this._content];
  }

  get setActions() {
    return [...this._setActions];
  }

  get factPaths() {
    return [...this._factPaths];
  }

  // Returns true if current screen is available based on conditions
  isAvailable(factGraph: FactGraph, collectionId: string | null): boolean {
    // If this screen is part of an auto-iterating collection loop, check first that it has members
    const autoIteratingLoopHasMembers = this.collectionLoop?.autoIterate
      ? !!this.collectionContext &&
        (convertCollectionToArray(factGraph.get(this.collectionContext).get) as string[]).length > 0
      : true;

    if (!autoIteratingLoopHasMembers) {
      return false;
    }

    // If there are explicit conditions, check those conditions
    const conditionsPass = this.conditions
      ? this.conditions.every((rawCondition) => {
          const condition = new Condition(rawCondition);
          return condition.evaluate(factGraph, collectionId);
        })
      : true;

    return conditionsPass;
  }
}
