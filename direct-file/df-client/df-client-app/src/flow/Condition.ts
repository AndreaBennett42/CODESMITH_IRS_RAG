/* eslint-disable eqeqeq */
import { assertNever } from 'assert-never';
import * as sfg from '@irs/js-factgraph-scala';
import { Path } from './Path.js';
import { Path as FDPath } from '../fact-dictionary/Path.js';
import { submissionBlockingFacts } from '../fact-dictionary/submissionBlockingFacts.js';
import { RootState, store } from '../redux/store.js';
import { stripNonNumeric } from '../misc/misc.js';
import { DataImportProfile } from '../redux/slices/data-import/dataImportProfileTypes.js';

export type ConditionOperator =
  | `isTrue`
  | `isTrueAndComplete`
  | `isTrueOrIncomplete`
  | `isFalse`
  | `isFalseAndComplete`
  | `isFalseOrIncomplete`
  | `isIncomplete`
  | `isComplete`;
export type ConditionString = `${FDPath}` | 'experimental' | 'submissionBlockingFactsAreFalse' | 'isEssarSigningPath';
export type RawCondition =
  | ConditionString
  | { operator?: ConditionOperator; condition: ConditionString }
  | DataImportRawCondition;

type DataImportConditionOperator = 'isTrue' | 'isFalse' | 'isUnknown';
type DataImportRawCondition = {
  condition: 'data-import';
  section:
    | `about-you`
    | `ip-pin-taxpayer-has-ip-pin`
    | `form-w2s`
    | `has-one-form-w2`
    | `has-multiple-forms-w2`
    | `has-one-1099-int`
    | `has-multiple-1099-ints`
    | `1099-ints`
    | `has-at-least-one-1095-a`
    | `has-no-1095-a`;
  operator?: DataImportConditionOperator;
};

export function rawConditionToString(condition: RawCondition) {
  if (typeof condition === `string`) {
    return condition;
  } else {
    return condition.operator + `:` + condition.condition;
  }
}

export class Condition {
  readonly innerCondition:
    | FeatureFlagCondition
    | PathCondition
    | RejectableFactCondition
    | EssarSigningCondition
    | DataImportCondition;
  constructor(private readonly rawCondition: RawCondition) {
    if (
      rawCondition === `experimental` ||
      // eslint-disable-next-line eqeqeq
      (typeof rawCondition != `string` && rawCondition.condition === `experimental`)
    ) {
      this.innerCondition = new FeatureFlagCondition();
    } else if (
      rawCondition === `isEssarSigningPath` ||
      // eslint-disable-next-line eqeqeq
      (typeof rawCondition != `string` && rawCondition.condition === `isEssarSigningPath`)
    ) {
      const pathOperator = typeof rawCondition === `object` && rawCondition.operator ? rawCondition.operator : `isTrue`;
      this.innerCondition = new EssarSigningCondition(pathOperator);
    } else if (
      rawCondition === `submissionBlockingFactsAreFalse` ||
      // eslint-disable-next-line eqeqeq
      (typeof rawCondition != `string` && rawCondition.condition === `submissionBlockingFactsAreFalse`)
    ) {
      this.innerCondition = new RejectableFactCondition();
      // eslint-disable-next-line eqeqeq
    } else if (typeof rawCondition != `string` && rawCondition.condition === `data-import`) {
      const operator = rawCondition.operator ?? `isTrue`;
      this.innerCondition = new DataImportCondition(operator, rawCondition.section);
    } else {
      const conditionString = typeof rawCondition === `string` ? rawCondition : rawCondition.condition;
      const pathOperator = typeof rawCondition === `object` && rawCondition.operator ? rawCondition.operator : `isTrue`;
      this.innerCondition = new PathCondition(conditionString as FDPath, pathOperator);
    }
  }

  evaluate(factGraph: sfg.FactGraph, collectionId: string | null): boolean {
    return this.innerCondition.evaluate(factGraph, collectionId);
  }

  toString() {
    return rawConditionToString(this.rawCondition);
  }
}

class FeatureFlagCondition {
  public factPath: undefined;
  evaluate() {
    return window.location.href.indexOf(`experimental`) != -1;
  }
}

class RejectableFactCondition {
  public factPath: undefined;
  evaluate(factGraph: sfg.FactGraph) {
    // this is every fact marked <BlockSubmissionOnTrue /> in the fact dictionary
    // If any of them are true, this condition evaluates to false
    return submissionBlockingFacts.every((path) => {
      const result = factGraph.get(Path.concretePath(path, null));
      // We do not check for completeness here -- some submission blocking facts (at least `/flowIsKnockedOut`)
      // use placeholder values, and we want to allow people through the tax return.
      return result.hasValue && result.get == false;
    });
  }
}

class PathCondition {
  readonly operator: ConditionOperator;
  public readonly factPath: FDPath;

  constructor(rawCondition: FDPath, operator: ConditionOperator) {
    this.operator = operator;
    this.factPath = rawCondition as FDPath;
  }

  evaluate(factGraph: sfg.FactGraph, collectionId: string | null): boolean {
    if (Path.isAbstract(this.factPath) && collectionId === null) {
      return false;
    }
    const fact = factGraph.get(Path.concretePath(this.factPath, collectionId));
    if (this.operator == `isTrue`) {
      return fact.hasValue && !!fact.get;
    } else if (this.operator == `isTrueAndComplete`) {
      return fact.complete && !!fact.get;
    } else if (this.operator == `isTrueOrIncomplete`) {
      return !fact.complete || !!fact.get;
    } else if (this.operator == `isFalse`) {
      return fact.hasValue && !fact.get;
    } else if (this.operator == `isFalseAndComplete`) {
      return fact.complete && !fact.get;
    } else if (this.operator == `isFalseOrIncomplete`) {
      return !fact.complete || !fact.get;
    } else if (this.operator == `isComplete`) {
      return fact.complete;
    } else if (this.operator == `isIncomplete`) {
      return !fact.complete;
    } else {
      return assertNever(this.operator);
    }
  }
}

// Allows us to enable or disable ESSAR-specific behavior using conditions eg:
//  displayOnlyIf={{ operator: `isFalseOrIncomplete`, condition: `isEssarSigningPath` }}
class EssarSigningCondition {
  public readonly factPath: FDPath;
  readonly operator: ConditionOperator;

  constructor(operator: ConditionOperator) {
    this.factPath = `/isLegacySigningPath`;
    this.operator = operator;
  }

  evaluate(factGraph: sfg.FactGraph, _collectionId: string | null): boolean {
    let valueWhenEssarDisabled;
    if (this.operator == `isTrue`) {
      valueWhenEssarDisabled = false;
    } else if (this.operator == `isTrueAndComplete`) {
      valueWhenEssarDisabled = false;
    } else if (this.operator == `isTrueOrIncomplete`) {
      valueWhenEssarDisabled = false;
    } else if (this.operator == `isFalse`) {
      valueWhenEssarDisabled = true;
    } else if (this.operator == `isFalseAndComplete`) {
      valueWhenEssarDisabled = true;
    } else if (this.operator == `isFalseOrIncomplete`) {
      valueWhenEssarDisabled = true;
    } else if (this.operator == `isComplete`) {
      valueWhenEssarDisabled = true;
    } else if (this.operator == `isIncomplete`) {
      valueWhenEssarDisabled = false;
    } else {
      return assertNever(this.operator);
    }

    const essarSigningEnabled = import.meta.env.VITE_ENABLE_ESSAR_SIGNING == `true`;
    const isLegacySigningPath = new PathCondition(this.factPath, this.operator).evaluate(factGraph, null);
    const isEssarSigningPath = !isLegacySigningPath;

    if (essarSigningEnabled == false) {
      return valueWhenEssarDisabled;
    } else {
      return isEssarSigningPath;
    }
  }
}

function getW2Result(state: RootState): DataImportProfile[`data`][`w2s`] | `not-loaded` {
  const dataImportState = state.dataImportProfile.data;
  if (dataImportState.status === `complete`) {
    const w2sResult = dataImportState.profile.data.w2s;
    if (w2sResult.state === `success`) {
      return w2sResult;
    }
  }
  return `not-loaded`;
}

function getAboutYouBasicResult(state: RootState): DataImportProfile[`data`][`aboutYouBasic`] | `not-loaded` {
  const dataImportState = state.dataImportProfile.data;
  if (dataImportState.status === `complete`) {
    const aboutYouBasicResult = dataImportState.profile.data.aboutYouBasic;
    if (aboutYouBasicResult.state === `success`) {
      return aboutYouBasicResult;
    }
  }
  return `not-loaded`;
}

function getHasW2Data(state: RootState): true | 'not-loaded' {
  const data = getW2Result(state);
  if (data === `not-loaded`) {
    return `not-loaded`;
  }
  if (data.state !== `success`) {
    return `not-loaded`;
  }
  if (data.payload === null) {
    return `not-loaded`;
  }

  if (data.payload.length === 0) {
    return `not-loaded`;
  }

  return true;
}

function get1099IntResult(state: RootState): DataImportProfile[`data`][`interestIncome`] | `not-loaded` {
  const dataImportState = state.dataImportProfile.data;
  if (dataImportState.status === `complete`) {
    const interestIncomeResult = dataImportState.profile.data.interestIncome;
    if (interestIncomeResult.state === `success`) {
      return interestIncomeResult;
    }
  }
  return `not-loaded`;
}

function getHas1099IntData(state: RootState): true | 'not-loaded' {
  const data = get1099IntResult(state);
  if (data === `not-loaded`) {
    return `not-loaded`;
  }
  if (data.state !== `success`) {
    return `not-loaded`;
  }
  if (data.payload === null) {
    return `not-loaded`;
  }
  if (data.payload.length === 0) {
    return `not-loaded`;
  }
  return true;
}

function getHasAboutYouBasicData(state: RootState): true | 'not-loaded' {
  const data = getAboutYouBasicResult(state);
  const result = data !== `not-loaded` && data.state === `success` ? data.payload !== null : false;
  return result || `not-loaded`;
}

class DataImportCondition {
  public readonly factPath: FDPath;
  readonly operator: DataImportConditionOperator;
  readonly section: DataImportRawCondition['section'];

  constructor(operator: DataImportConditionOperator, section: DataImportRawCondition['section']) {
    this.factPath = `/flowFalse`;
    this.operator = operator;
    this.section = section;
  }

  getUnderlyingValue(factGraph: sfg.FactGraph, collectionId: string | null): boolean | 'not-loaded' {
    const reduxState = store.getState();
    if (this.section === `about-you`) {
      const result = getHasAboutYouBasicData(reduxState);

      if (result === `not-loaded`) {
        return false;
      }
      return result;
    } else if (this.section === `ip-pin-taxpayer-has-ip-pin`) {
      const isResubmittingPath = Path.concretePath(`/isResubmitting`, collectionId);
      const isResubmitting = factGraph.get(isResubmittingPath).complete ? factGraph.get(isResubmittingPath).get : false;
      if (isResubmitting) {
        return `not-loaded`;
      }
      // At the time of writing this code, all imported facts are initialized in a data preview
      // screen, which gives the flow a chance to later condition on whether or not those facts
      // are present.
      //
      // In the case of IP PIN, we have a unique requirement: we need to skip the IP PIN screen
      // entirely if data import says that the user does not have an IP PIN. While it's simple
      // enough to condition on data import (that's what this file does), without a data preview
      // to initialize this fact, there's no place to set the fact to false *before* rendering
      // the IP PIN in the flow. We can't do it in the <IpPin> component, because once the flow
      // starts rendering that component, it's already in the user's navigation, which creates a
      // huge number of problems.
      //
      // A better place to do this would be during the so-called magic-screen where Data Import
      // "happens" and then is integrated into the fact graph. This is a temporary solution
      // until such a screen exists.
      // https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/12293
      const data = reduxState.dataImportProfile.data;
      if (data.status === `complete` && data.profile.data.ipPin.state === `success`) {
        const payload = data.profile.data.ipPin.payload;
        const hasIpPinPath = Path.concretePath(`/filers/*/hasIpPin`, collectionId);
        const identityPinPath = Path.concretePath(`/filers/*/identityPin`, collectionId);
        if (payload.hasIpPin === true) {
          const result = sfg.IpPinFactory(stripNonNumeric(payload.pin));
          if (result.isRight) {
            factGraph.set(hasIpPinPath, true);
            factGraph.set(identityPinPath, result.right);
            return true;
          }
        } else {
          factGraph.set(hasIpPinPath, false);
          return false;
        }
      }
      return `not-loaded`;
    } else if (this.section === `form-w2s`) {
      const result = getHasW2Data(reduxState);
      if (result === `not-loaded`) {
        return false;
      }
      return result;
    } else if (this.section === `has-one-form-w2`) {
      const result = getW2Result(reduxState);
      if (result === `not-loaded`) {
        return false;
      }
      if (result.state === `success`) {
        return result.payload.length === 1;
      }
      return false;
    } else if (this.section === `has-multiple-forms-w2`) {
      const result = getW2Result(reduxState);
      if (result === `not-loaded`) {
        return false;
      }
      if (result.state === `success`) {
        return result.payload.length > 1;
      }
      return false;
    } else if (this.section === `has-one-1099-int`) {
      const result = get1099IntResult(reduxState);
      if (result === `not-loaded`) {
        return false;
      }
      if (result.state === `success`) {
        return result.payload.length === 1;
      }
      return false;
    } else if (this.section === `has-multiple-1099-ints`) {
      const result = get1099IntResult(reduxState);
      if (result === `not-loaded`) {
        return false;
      }
      if (result.state === `success`) {
        return result.payload.length > 1;
      }
      return false;
    } else if (this.section === `1099-ints`) {
      const result = getHas1099IntData(reduxState);
      if (result === `not-loaded`) {
        return false;
      }
      return result;
    } else if (this.section === `has-at-least-one-1095-a`) {
      const data = reduxState.dataImportProfile.data;
      if (data.status === `complete` && data.profile.data.f1095A.state === `success`) {
        if (data.profile.data.f1095A.payload.has1095A === true) {
          return true;
        }
      }
      return `not-loaded`;
    } else if (this.section === `has-no-1095-a`) {
      const data = reduxState.dataImportProfile.data;
      if (data.status === `complete` && data.profile.data.f1095A.state === `success`) {
        if (data.profile.data.f1095A.payload.has1095A === false) {
          return true;
        }
      }
      return `not-loaded`;
    } else {
      return assertNever(this.section);
    }
  }

  evaluate(factGraph: sfg.FactGraph, collectionId: string | null): boolean {
    const underlyingValue = this.getUnderlyingValue(factGraph, collectionId);
    if (this.operator === `isTrue`) {
      return underlyingValue === true;
    } else if (this.operator === `isFalse`) {
      return underlyingValue === false;
    } else if (this.operator === `isUnknown`) {
      return underlyingValue == `not-loaded`;
    } else if (this.operator === `isUnknownOrResubmitting`) {
      return underlyingValue == `not-loaded`;
    }

    return false;
  }
}
