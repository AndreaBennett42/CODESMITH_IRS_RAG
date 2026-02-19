import * as sfg from '@irs/js-factgraph-scala';
import { CURRENT_TAX_YEAR } from '../constants/taxConstants.js';
import { Path } from '../flow/Path.js';
import { getFieldsToClearOnFilingStatusChange } from '../utils/fieldClearing.js';
import { FlowConfig } from '../flow/flowConfig.js';
import { wrappedFacts } from '../fact-dictionary/generated/wrappedFacts.js';
import { AbsolutePath, Path as FDPath } from '../fact-dictionary/Path.js';
import { getEnumOptions } from '../hooks/useEnumOptions.js';

/**
 * Deletes all currently selected facts for the specific FactSelect instance
 * @param factGraph
 * @param factPath
 * @param collectionId
 */
function clearSelectedFacts(factGraph: sfg.FactGraph, factPath: AbsolutePath, collectionId: string) {
  const { values: options } = getEnumOptions(factGraph, factPath, collectionId);

  const pathPrefix = factPath.split(`/`).slice(0, -1).join(`/`);

  if (options === undefined) {
    return;
  } else {
    for (const option of options) {
      const selectableFactPath = Path.concretePath(`${pathPrefix}/${option}` as AbsolutePath, collectionId);

      factGraph.delete(selectableFactPath);
    }
  }
}

function normalizeW2s(factGraph: sfg.FactGraph, dirtyFacts: sfg.ConcretePath[]) {
  const collectionPath = Path.concretePath(`/formW2s`, null);
  const collectionResult = factGraph.get(collectionPath);

  let didClear = false;
  if (collectionResult.complete) {
    const collectionIds: string[] = sfg.scalaListToJsArray<string>(collectionResult.get.getItemsAsStrings());

    for (const collectionId of collectionIds) {
      for (const boxNumber of [`12`, `14`] as const) {
        const gatePath = Path.concretePath(`/formW2s/*/writableHasBox${boxNumber}Codes`, collectionId);

        if (dirtyFacts.includes(gatePath)) {
          const gateFact: sfg.FactGraphResult<boolean> = factGraph.get(gatePath);

          // Clear only if the user has explicitly responded with "no"
          const shouldClear = gateFact.complete && !gateFact.get;

          if (shouldClear) {
            didClear = true;
            clearSelectedFacts(factGraph, `/formW2s/*/writableBox${boxNumber}Code`, collectionId);
          }
        }
      }
    }
  }

  // eslint-disable-next-line df-rules/no-factgraph-save
  if (didClear) factGraph.save();
}

const CARE_PROVIDER_W2_REFERENCE_PATH = `/cdccCareProviders/*/writableEmployerWhoFurnishedCare`;
const CARE_PROVIDER_HAS_W2_REFERENCE_PATH = `/cdccCareProviders/*/hasW2Employer`;
function clearDanglingW2References(factGraph: sfg.FactGraph, dirtyFacts: sfg.ConcretePath[]) {
  if (dirtyFacts.some((fact) => fact.startsWith(`/formW2s`))) {
    const currentFormW2sFact = factGraph.get(Path.concretePath(`/formW2s`, null));
    const currentCareProvidersFact = factGraph.get(Path.concretePath(`/cdccCareProviders`, null));

    if (!(currentFormW2sFact.hasValue && currentCareProvidersFact.hasValue)) return;

    const currentFormW2Ids = sfg.scalaListToJsArray(currentFormW2sFact.get.getItemsAsStrings());
    const currentCareProviderIds = sfg.scalaListToJsArray(currentCareProvidersFact.get.getItemsAsStrings());

    if (currentFormW2Ids.length === 0 || currentCareProviderIds.length === 0) return;

    let didChange = false;

    // Find any care providers that reference a previously removed W2 item
    for (const careProviderId of currentCareProviderIds) {
      const hasW2ReferencePath = Path.concretePath(CARE_PROVIDER_HAS_W2_REFERENCE_PATH, careProviderId);
      const w2ReferencePath = Path.concretePath(CARE_PROVIDER_W2_REFERENCE_PATH, careProviderId);

      const maybeW2ReferenceFact = factGraph.get(w2ReferencePath);
      if (maybeW2ReferenceFact.hasValue && !currentFormW2Ids.includes(maybeW2ReferenceFact.get.idString)) {
        factGraph.delete(hasW2ReferencePath);
        factGraph.delete(w2ReferencePath);
        didChange = true;
      }
    }

    // eslint-disable-next-line df-rules/no-factgraph-save
    if (didChange) factGraph.save();
  }
}

/**
 * The scala.js FactGraph is hard to extend -- scala.js creates a weird object that doesn't meet
 * many of my expectations around how a javascript object should work. Therefore, instead of
 * extending the object we add javascript functionality by creating a wrapper, and we
 * can use the wrapper like a normal javascript object.
 *
 */
export class InterceptingFactGraph implements sfg.FactGraph {
  public sfgGraph: sfg.FactGraph;
  flow?: FlowConfig;
  allRoutes: string[] = [];
  reviewScreenIndex = -1;
  reviewAndRoutesComingAfterInTheFlow: Set<string> | undefined;
  previousFinalTaxAmount: string | undefined = undefined;

  constructor(existingFacts = {}, flow?: FlowConfig) {
    this.flow = flow;
    this.initializeFlowDependentProperties();
    this.dirtyFacts = [];

    const meta = new sfg.DigestMetaWrapper(CURRENT_TAX_YEAR).toNative();

    // TODO: Write a type for the digest stuff
    const facts = wrappedFacts.map((fact) =>
      sfg.DigestNodeWrapperFactory.toNative(
        new sfg.DigestNodeWrapper(fact.path, fact.writable, fact.derived, fact.placeholder)
      )
    );

    const config = sfg.FactDictionaryConfig.create(meta, facts);

    const factDictionary = sfg.FactDictionaryFactory.fromConfig(config);

    const existingFactJsonString = JSON.stringify(existingFacts);
    const persister = sfg.JSPersister.create(existingFactJsonString);

    const factGraph = sfg.GraphFactory.apply(factDictionary, persister);

    // TODO: we should have a less nasty way of debugging these.
    // Right now, the last fact graph to set up will win.
    globalThis.debugFactGraph = this;
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    globalThis.debugScalaFactGraphLib = sfg;
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    globalThis.debugFactGraphMeta = meta;
    globalThis.debugFacts = facts;
    this.sfgGraph = factGraph;
  }

  private dirtyFacts: sfg.ConcretePath[];

  private initializeFlowDependentProperties(): void {
    if (!this.flow) {
      return;
    }
    this.allRoutes = Array.from(this.flow.screensByRoute.keys());
    this.reviewScreenIndex = this.allRoutes.indexOf(`/flow/complete/review/review`);
    this.reviewAndRoutesComingAfterInTheFlow = new Set(
      this.allRoutes.slice(this.reviewScreenIndex).filter((route) => !route.includes(`knockout`))
    );
  }

  public delete(path: sfg.ConcretePath) {
    this.dirtyFacts.push(path);
    return this.sfgGraph.delete(path);
  }
  public getDictionary() {
    return this.sfgGraph.getDictionary();
  }
  public set(path: sfg.ConcretePath, value: unknown) {
    this.dirtyFacts.push(path);
    return this.sfgGraph.set(path, value);
  }

  /**
   * Our get method handles the following scenarios:
   * - where the final tax amount has changed, and an alert should be displayed.
   * @param path The path of the requested fact.
   * @returns The requested fact.
   */
  public get(path: sfg.ConcretePath) {
    /**
     *  An amount changed alert is only displayed if all of the following are true:
     * - The user has already visited the amount screen
     * - The amount is different than the previous amount
     *  */
    const FINAL_TAX_AMOUNT_FACT = Path.concretePath(`/finalTaxAmount`, null);
    if (path === FINAL_TAX_AMOUNT_FACT) {
      // sfgGraph is called directly to avoid recursively triggering this code.
      const nextFinalTaxAmount = this.sfgGraph.get(path);

      const FINAL_TAX_AMOUNT_SEEN_FACT = Path.concretePath(`/flowHasSeenAmount`, null);
      const hasSeenFinalTaxAmount = this.get(FINAL_TAX_AMOUNT_SEEN_FACT);

      // Set the tax amount for the first time if it is undefined.
      if (nextFinalTaxAmount.complete && this.previousFinalTaxAmount === undefined) {
        this.previousFinalTaxAmount = nextFinalTaxAmount.get?.toString();
      }

      // Detect a change from the previously displayed amount.
      if (
        hasSeenFinalTaxAmount.complete &&
        hasSeenFinalTaxAmount.get &&
        nextFinalTaxAmount.complete &&
        nextFinalTaxAmount.get?.toString() !== this.previousFinalTaxAmount
      ) {
        this.set(Path.concretePath(`/flowHasAmountChanged`, null), true);
        this.previousFinalTaxAmount = nextFinalTaxAmount.get?.toString();
        this.save();
      }
    }
    return this.sfgGraph.get(path);
  }

  public getFact(path: sfg.ConcretePath): sfg.Fact {
    return this.sfgGraph.getFact(path);
  }

  // Helper function for getting the value using path + collectionId
  public getValue(path: FDPath, collectionId: string | null) {
    return this.sfgGraph.get(Path.concretePath(path, collectionId)).get.toString();
  }

  /**
   *  Our save method handles the following scenarios:
   * - where a person has changed their filing status, and we need to delete some of their data.
   * - where we've autocorrected the person's payment method while on the paper path, they later
   * resolve the paper path warning, and we need to reset the payment method.
   */
  public save() {
    const currentPathMatch = window.location.pathname.match(/\/flow.*/);
    const currentPath = currentPathMatch ? currentPathMatch[0] : ``;
    const shouldMarkReviewUnseen =
      this.reviewAndRoutesComingAfterInTheFlow && !this.reviewAndRoutesComingAfterInTheFlow.has(currentPath);
    if (shouldMarkReviewUnseen) {
      this.set(Path.concretePath(`/hasSeenReviewScreen`, null), false);
    }
    const FILING_STATUS_FACT = `/filingStatus`;
    const MFJ_OPTION = `marriedFilingJointly`;
    const MFS_CONSIDERED_UNMARRIED_FACT = `/mfsButEligibleForCdcc`;
    const PF_MAX_DEP_CARE_EXCLUSION_FACT = `/writablePrimaryFilerDependentCarePlanMaximum`;
    const filingStatusBeforeSaveResult = this.get(Path.concretePath(FILING_STATUS_FACT, null));
    const mfsConsideredUnmarriedBeforeSaveResult = this.get(Path.concretePath(MFS_CONSIDERED_UNMARRIED_FACT, null));

    const IS_PAPER_PATH_FACT = `/isPaperPath`;
    const PAYMENT_METHOD_FACT = `/payViaAch`;
    const PAYMENT_METHOD_WAS_AUTOCORRECTED_FACT = `/flowHasSeenPaymentMethodAutocorrect`;
    const PAYMENT_METHOD_ASSERTION_FACT = `/flowHasSeenPaymentPaperPathAssertion`;
    const isPaperPathBeforeSaveResult = this.get(Path.concretePath(IS_PAPER_PATH_FACT, null));
    const paymentMethodWasAutoCorrectedBeforeSaveResult = this.get(
      Path.concretePath(PAYMENT_METHOD_WAS_AUTOCORRECTED_FACT, null)
    );

    // eslint-disable-next-line df-rules/no-factgraph-save
    const ret = this.sfgGraph.save();

    const filingStatusAfterSaveResult = this.get(Path.concretePath(FILING_STATUS_FACT, null));

    const isPaperPathAfterSaveResult = this.get(Path.concretePath(IS_PAPER_PATH_FACT, null));

    const mfsConsideredUnmarriedAfterSaveResult = this.get(Path.concretePath(MFS_CONSIDERED_UNMARRIED_FACT, null));

    const clearEmployerExclusionField =
      mfsConsideredUnmarriedBeforeSaveResult.complete &&
      mfsConsideredUnmarriedAfterSaveResult.complete &&
      // eslint-disable-next-line eqeqeq
      mfsConsideredUnmarriedBeforeSaveResult.get.toString() != mfsConsideredUnmarriedAfterSaveResult.get.toString();

    const clearFilingStatusFields =
      filingStatusBeforeSaveResult.complete &&
      filingStatusAfterSaveResult.complete &&
      // eslint-disable-next-line eqeqeq
      filingStatusAfterSaveResult.get.toString() != filingStatusBeforeSaveResult.get.toString() &&
      (filingStatusAfterSaveResult.get.getValue() === MFJ_OPTION ||
        filingStatusBeforeSaveResult.get.getValue() === MFJ_OPTION);

    const clearPaymentMethodFields =
      isPaperPathBeforeSaveResult.complete &&
      isPaperPathBeforeSaveResult.get &&
      isPaperPathAfterSaveResult.complete &&
      !isPaperPathAfterSaveResult.get &&
      paymentMethodWasAutoCorrectedBeforeSaveResult.complete &&
      paymentMethodWasAutoCorrectedBeforeSaveResult.get;

    if (clearFilingStatusFields) {
      const fieldsToClear = getFieldsToClearOnFilingStatusChange(this);
      for (const field of fieldsToClear) {
        debugFactGraph.delete(field);
      }
      // eslint-disable-next-line df-rules/no-factgraph-save
      return this.sfgGraph.save();
    }

    if (clearPaymentMethodFields) {
      debugFactGraph.delete(Path.concretePath(PAYMENT_METHOD_FACT, null));
      debugFactGraph.delete(Path.concretePath(PAYMENT_METHOD_WAS_AUTOCORRECTED_FACT, null));
      debugFactGraph.delete(Path.concretePath(PAYMENT_METHOD_ASSERTION_FACT, null));
      // eslint-disable-next-line df-rules/no-factgraph-save
      return this.sfgGraph.save();
    }

    if (clearEmployerExclusionField) {
      debugFactGraph.delete(Path.concretePath(PF_MAX_DEP_CARE_EXCLUSION_FACT, null));
      // eslint-disable-next-line df-rules/no-factgraph-save
      return this.sfgGraph.save();
    }

    normalizeW2s(this.sfgGraph, this.dirtyFacts);
    clearDanglingW2References(this.sfgGraph, this.dirtyFacts);

    this.dirtyFacts = [];

    return ret;
  }
  public toJSON() {
    return this.sfgGraph.toJSON();
  }
  public toStringDictionary() {
    return this.sfgGraph.toStringDictionary();
  }
  public toJson() {
    return this.sfgGraph.toJson();
  }
  public explainAndSolve(path: sfg.ConcretePath) {
    return this.sfgGraph.explainAndSolve(path);
  }
  public download(filename: string | undefined) {
    downloadFactGraph(this.sfgGraph, filename);
  }
  public checkPersister() {
    return this.sfgGraph.checkPersister();
  }
}

function downloadFactGraph(factGraph: sfg.FactGraph, filename: string | undefined) {
  if (!filename) {
    const timestamp = new Date().toISOString();
    filename = `fact-graph-${timestamp}.json`;
  }
  const jsonString = factGraph.toJson();
  downloadTextFile(jsonString, filename, `application/json`);
}

function downloadTextFile(text: string, filename: string, mimeType: string) {
  const blob = new Blob([text], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement(`a`);
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
