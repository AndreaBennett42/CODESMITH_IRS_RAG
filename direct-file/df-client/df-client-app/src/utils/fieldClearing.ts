import { ConcretePath, FactGraph, scalaListToJsArray } from '@irs/js-factgraph-scala';
import { createFlowConfig, FlowConfig } from '../flow/flowConfig.js';
import flowNodes from '../flow/flow.js';
import { Path } from '../flow/Path.js';
import { Path as FDPath } from '../fact-dictionary/Path.js';

// I do not know a way to keep this up to date
// but anytime there's a fact in the fact dictionary that
// varies meaning based on having a spouse, we should delete it.
export const CLEARABLE_DEPENDENT_FACTS: FDPath[] = [
  `/familyAndHousehold/*/relationshipCategory`,
  `/familyAndHousehold/*/siblingRelationship`,
  `/familyAndHousehold/*/childRelationship`,
  `/familyAndHousehold/*/parentalRelationship`,
  `/familyAndHousehold/*/inlawRelationship`,
  `/familyAndHousehold/*/hasOtherBiologicalOrAdoptiveParent`,
  `/familyAndHousehold/*/whichParentNotClaiming`,
  `/familyAndHousehold/*/tpAgiHigherThanOtherParent`,
  `/familyAndHousehold/*/tpPaidMostOfHomeUpkeep`,
  `/familyAndHousehold/*/tpPaidMostOfParentHomeUpkeep`,
  `/familyAndHousehold/*/writableQrSupportTest`,
  `/familyAndHousehold/*/writableCouldBeQualifyingChildOfAnother`,
  `/familyAndHousehold/*/writablePotentialClaimerMustFile`,
  `/familyAndHousehold/*/writablePotentialClaimerDidFile`,
  `/familyAndHousehold/*/writablePotentialClaimerFiledOnlyForRefund`,
  `/familyAndHousehold/*/cdccHadExpensesPaidToQualifyingProvider`,
  `/familyAndHousehold/*/cdccHasDependentCareExpenses`,

  `/familyAndHousehold/*/writableCdccQualifyingExpenseAmount`,
];

export function getAllPathsFor1095A(flow: FlowConfig) {
  const form1095ALoop = flow.collectionLoopsByName.get(`/1095As`);
  const listOfAllForm1095AFacts = form1095ALoop?.screens.flatMap((sc) => sc.factPaths) || [];
  // Add to ensure the collection itself is deleted
  listOfAllForm1095AFacts.push(`/1095As`);
  return listOfAllForm1095AFacts;
}

/**
 * When a user changes their filing status, we have to clear any field that may have a different meaning
 * when the filing status is different. This includes:
 *
 * - All credits (since the questions "were you disqualified/were you or your MFJ spouse disqualified") are different
 * - All deductions -- since the spouse may have loans or educator expenses
 * - A few questions in the dependents section related to custody ("Did they live with you or your spouse")
 */
export function getFieldsToClearOnFilingStatusChange(factGraph: FactGraph): ConcretePath[] {
  const flow = createFlowConfig(flowNodes);
  const deductionsSubcategory = flow.subcategoriesByRoute.get(`/flow/credits-and-deductions/deductions`);
  const creditsSubcategory = flow.subcategoriesByRoute.get(`/flow/credits-and-deductions/credits`);
  if (!deductionsSubcategory || !creditsSubcategory) {
    throw new Error(`Credits and deductions subcategories must exist`);
  }

  // We skip abstract paths here since the only collections in these sections
  // A corresponding test will break if someone adds a collection context that A.M. didn't expect
  // on January 9, 2024. In that case, you may need to add a collection context like we do with dependents below.

  // We have since added collections within the credits section. Since credit eligibility is usually dependent on
  // filing status we'll generally want to clear out the associated collections.
  const deductionFacts = deductionsSubcategory.screens
    .flatMap((sc) => sc.factPaths)
    .filter((p) => !Path.isAbstract(p)) as ConcretePath[];
  const creditsFacts = creditsSubcategory.screens
    .flatMap((sc) => sc.factPaths)
    .filter((p) => !Path.isAbstract(p)) as ConcretePath[];

  const listOfAllForm1095AFacts = getAllPathsFor1095A(flow);

  const dependentsResult = factGraph.get(`/familyAndHousehold` as ConcretePath);

  const dependents: string[] = dependentsResult.complete
    ? scalaListToJsArray(dependentsResult.get.getItemsAsStrings())
    : [];
  const dependentFacts = dependents.flatMap((d) => CLEARABLE_DEPENDENT_FACTS.map((f) => Path.concretePath(f, d)));

  const form1095AResults = factGraph.get(`/1095As` as ConcretePath);
  const form1095As: string[] = form1095AResults.complete
    ? scalaListToJsArray(form1095AResults.get.getItemsAsStrings())
    : [];
  const form1095Facts = form1095As.flatMap((d) => listOfAllForm1095AFacts.map((f) => Path.concretePath(f, d)));

  return [...deductionFacts, ...creditsFacts, ...dependentFacts, ...form1095Facts];
}
