import type { ReactElement, FC } from 'react';
import type { RawCondition } from './Condition.js';
import type { DataPreviewDeclaration, ScreenContentNode } from './ContentDeclarations.js';
import type { IconName } from '../components/IconDisplay/IconDisplay.js';
import type { ItemAssertion, SubSubcategoryBorderStyle } from './flowConfig.js';
import type { AbsolutePath, Path, WritableAbsolutePath } from '../fact-dictionary/Path.js';
import { DataItemConfig } from '../components/checklist/ChecklistSubcategory/DataReveal.js';
import { HeadingLevel } from '@trussworks/react-uswds';

type OneOrArray<T> = T | T[];
interface FlowDeclaration {
  children: OneOrArray<GateNode | CategoryNode>;
}
export type FlowNode = ReactElement<FlowDeclaration>;
interface GateDeclaration {
  children: OneOrArray<CategoryNode | SubcategoryNode | GateNode | ScreenNode>;
  condition: RawCondition;
}
export type GateNode = ReactElement<GateDeclaration>;
export interface CategoryDeclaration {
  route: string;
  children: OneOrArray<SubcategoryNode | GateNode>;
}

export interface SubcategoryDeclaration {
  route: string;
  completeIf: RawCondition | RawCondition[];
  collectionContext?: Path;
  skipDataView?: true;
  isSignAndSubmit?: true;
  /**
   * A list of conditions for which *at least one* must be true to display.
   * NOTE: This condition is true if *any* conditions pass, unlike the
   * `conditions` property elsewhere which requires *all* facts to be true
   **/
  displayOnlyIf?: RawCondition | RawCondition[];
  lockFutureSectionsIfCollectionItemsIncomplete?: boolean;
  children: OneOrArray<GateNode | ScreenNode>;
  dataItems?: DataItemConfig[];
}

export interface AssertionDeclaration {
  type: `success` | `warning` | `inactive` | `info`;
  i18nKey: string;
  condition?: RawCondition;
  conditions?: RawCondition[];
  editRoute?: string;
}

export interface SubSubcategoryDeclaration {
  route: string;
  children: OneOrArray<GateNode | ScreenNode>;
  editable?: boolean;
  hidden?: boolean;
  completeIf?: RawCondition | RawCondition[]; // Needed for sscs that are a collection
  // maybe we need to detect completion?
  collectionContext?: Path;
  headingLevel?: HeadingLevel;
  borderStyle?: SubSubcategoryBorderStyle;
}

export type SubcategoryNode = ReactElement<SubcategoryDeclaration>;

export type AssertionNode = ReactElement<AssertionDeclaration>;

export type SubSubcategoryNode = ReactElement<SubSubcategoryDeclaration>;

export type CategoryNode = ReactElement<CategoryDeclaration>;
export type ScreenNode = ReactElement<ScreenDeclaration>;

export type AlertAggregatorType = `screen` | `sections`;
interface ScreenDeclaration {
  route: string;
  children: ScreenContentNode[];
  condition?: RawCondition;
  // NOTE: We're getting enough flags this might make sense to refactor into like a "role"
  // screens at some point, or different types of screen
  routeAutomatically?: boolean; // default is 'true'
  actAsDataView?: boolean;
  alertAggregatorType?: AlertAggregatorType;
  isKnockout?: boolean; // default is false
  hideBreadcrumbs?: boolean; // default is false
  // default is false, should only be used for inner collection items to ensure they are routed to their corresponding
  // data-view screens rather than their flow equivalents
  hasScreenRouteOverride?: boolean;
}

export interface CollectionLoopDeclaration {
  children: OneOrArray<ScreenNode>;
  autoIterate?: boolean;
  loopName: string; // instance of the loop
  collection: Path; // factname of collection
  iconName?: IconName;
  collectionItemCompletedCondition?: Path;
  hideCardLabel2Condition?: Path;
  isInner?: boolean; // indicates it is below subcat level
  donePath?: WritableAbsolutePath;
  isImportedFactPath?: AbsolutePath;
  importedFlowStartRoute?: string;
  importedFlowDonePath?: AbsolutePath;
  importedRouteOverride?: string;
  shouldSeeHubCompletionBtnsPath?: AbsolutePath;
  // Used to route a user if they're knocked out by completing
  // the collection loop
  knockoutRoute?: string;
  dataViewSections?: {
    i18nKey: string;
    i18nModalKey?: string;
    condition?: RawCondition;
    conditions?: RawCondition[];
    itemAssertions?: ItemAssertion[];
  }[];
  dataReveals?: DataItemConfig[];
}
export type CollectionLoopNode = ReactElement<CollectionLoopDeclaration>;

export type DataPreview = ReactElement<DataPreviewDeclaration>;

export type FlowChild =
  | GateNode
  | CategoryNode
  | SubcategoryNode
  | AssertionNode
  | ScreenNode
  | SubSubcategoryNode
  | CollectionLoopNode;

export const Flow: FC<FlowDeclaration> = () => null;
export const Gate: FC<GateDeclaration> = () => null;
export const Category: FC<CategoryDeclaration> = () => null;
export const Subcategory: FC<SubcategoryDeclaration> = () => null;
export const Assertion: FC<AssertionDeclaration> = () => null;
export const SubSubcategory: FC<SubSubcategoryDeclaration> = () => null;
export const Screen: FC<ScreenDeclaration> = () => null;
export const CollectionLoop: FC<CollectionLoopDeclaration> = () => null;

export const AllFlowTypes = {
  Flow,
  Gate,
  Category,
  Subcategory,
  Assertion,
  SubSubcategory,
  Screen,
  CollectionLoop,
};

export function isCategory(tree: FlowChild): tree is CategoryNode {
  return !Array.isArray(tree) && tree.type === Category;
}

export function isSubcategory(tree: FlowChild): tree is SubcategoryNode {
  return !Array.isArray(tree) && tree.type === Subcategory;
}

export function isAssertion(tree: FlowChild): tree is AssertionNode {
  return !Array.isArray(tree) && tree.type === Assertion;
}

export function isSubSubcategory(tree: FlowChild): tree is SubSubcategoryNode {
  return !Array.isArray(tree) && tree.type === SubSubcategory;
}

export function isScreen(tree: FlowChild): tree is ScreenNode {
  return !Array.isArray(tree) && tree.type === Screen;
}

export function isGate(tree: FlowChild): tree is GateNode {
  return !Array.isArray(tree) && tree.type === Gate;
}

export function isCollectionLoop(tree: FlowChild): tree is CollectionLoopNode {
  return !Array.isArray(tree) && tree.type === CollectionLoop;
}
