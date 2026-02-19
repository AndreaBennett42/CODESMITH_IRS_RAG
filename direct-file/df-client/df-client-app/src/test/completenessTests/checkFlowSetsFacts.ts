import {
  FactGraph,
  AddressFactory,
  UsPhoneNumberFactory,
  DayFactory,
  unwrapScalaOptional,
  scalaListToJsArray,
  ScalaList,
  EnumFactory,
  MultiEnumFactory,
  DollarFactory,
  TinFactory,
  EinFactory,
  IpPinFactory,
  PinFactory,
  BankAccountFactory,
} from '@irs/js-factgraph-scala';
import { assertNever } from 'assert-never';
import { expect } from 'vitest';
import { FactConfig } from '../../flow/ContentDeclarations.js';
import { createFlowConfig } from '../../flow/flowConfig.js';
import flowNodes from '../../flow/flow.js';
import { Path as CPath } from '../../flow/Path.js';
import getNextScreen from '../../screens/getNextScreen.js';
import { contentIsAvailableRequiredWritableFactConfig } from '../../flow/flowHelpers.js';
import { Path } from '../../fact-dictionary/Path.js';
import { setupFactGraphDeprecated } from '../setupFactGraph.js';

/**
 * checkFlowSetsFacts.ts
 *
 * Starting at a certain screen, check that every possible run through the next screens
 * in the flow will leave certain facts as complete before we next reach the checklist.
 *
 * For instance, you can check that any path through the interest loop will leave `/taxExemptInterest` complete
 *
 * It forks at any boolean or enum, and you can provide hardcoded
 * values to fork at other fact paths.
 *
 * Due to the combinatoric nature of forking like this, the test runs in NP-time. So please be careful
 * and pass in a small set of screens with a small set of booleans or enums. Circle will time out after 10
 * minutes if the run is too long.
 *
 * It would be good to write the following facts in this framework:
 * 1. Check that every `deductions` path leaves the filer with an /agi
 * 2. Check that every `credits` path leaves the filer with a tax amount (or maybe just eligibility)
 * 3. Check that every 'about you' and 'spouse' path leaves the filer with a standard deduction amount
 * 4. Check that all dependents complete eitcQualifyingChild, qssQualifyingPerson, hohQualifyingPerson,eligibleDependent
 */

export interface FlowSetFactInput {
  /* The starting screen route. The test will run until it hits a knockout or the checklist. */
  startingScreenRoute: string;
  /* The facts with which we initialize the fact graph */
  startingFactState: object;
  /* The facts that we expect to be complete at the end of the loop */
  expectedCompleteFacts: Path[];
  /* Starting fact state to check. E.g. if you want to check that your set of starting facts sets some
     derived facts (e.g. check that `/totalTax` is complete at the beginning of the section) you can do that here.
     The test runner will check that the path is complete, and has the stringified value listed in this parameter.
  */
  expectedStartingFactState: Partial<Record<Path, string>>;
  /* A collection id. Must already correspond to a collection created in `startingFactState` */
  collectionId: string;
  /* Hardcoded facts to sub in when the flow arrives at a certain path */
  factValuesToSet: Partial<Record<Path, unknown[]>>;
  /* Leave optional fields blank. Used to test that optional fields do not cause incomplete facts.
   * It would be better to branch on each fact for [defined, not defined] but that expands our runtime too far.
   */
  leaveOptionalFieldsBlank: boolean;
}

const flow = createFlowConfig(flowNodes);

interface HistoryFact {
  route: string;
  setFacts: { path: Path; value: unknown }[];
}

/**
 * Starting at a certain screen, check that every possible run through the next screens
 * in that flow will leave certain facts as complete before we next reach the checklist.
 *
 * Runs in NP-time relative to the number of enums and booleans in the specified flow,
 * so this will not work for long runs of screens that fork on a large number of enums and booleans.
 */
export function checkFlowSetsFacts(testInput: FlowSetFactInput) {
  const {
    startingScreenRoute,
    collectionId,
    startingFactState,
    factValuesToSet,
    expectedCompleteFacts,
    expectedStartingFactState,
    leaveOptionalFieldsBlank,
  } = testInput;
  const { factGraph: initialFactGraph } = setupFactGraphDeprecated(startingFactState);
  for (const entry of Object.entries(expectedStartingFactState)) {
    const result = initialFactGraph.get(CPath.concretePath(entry[0] as Path, collectionId));
    expect(result.complete, `At start of test, ${entry[0]} is not complete`).toBe(true);
    expect(result.get.toString(), `At start of test, ${entry[0]} is not equal to expected value`).toBe(entry[1]);
  }
  const screenGraphsToTest: { route: string; factGraphState: string; history: HistoryFact[] }[] = [];
  screenGraphsToTest.push({ route: startingScreenRoute, factGraphState: initialFactGraph.toJSON(), history: [] });
  let sawOnlyKnockouts = true;
  while (screenGraphsToTest.length > 0) {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const { route, factGraphState, history } = screenGraphsToTest.pop()!;
    const { factGraph } = setupFactGraphDeprecated(JSON.parse(factGraphState));
    getPotentialFactGraphsAfterScreen(
      route,
      factGraph,
      collectionId,
      factValuesToSet,
      leaveOptionalFieldsBlank
    ).forEach((branch) => {
      const { factGraph } = setupFactGraphDeprecated(JSON.parse(branch.factGraphJson));
      const screen = flow.screensByRoute.get(route);
      if (!screen) {
        throw new Error(`Not on a valid screen`);
      }
      const nextScreen = getNextScreen(screen, factGraph, collectionId, flow);
      if (!nextScreen) {
        throw new Error(`Broken`);
      }
      if (nextScreenIsKnockout(nextScreen.routable.fullRoute(nextScreen.collectionId))) {
        // do nothing. Knockouts may not set all facts for the terminal case.
      } else if (nextScreenIsTerminal(nextScreen.routable.fullRoute(nextScreen.collectionId))) {
        // eslint-disable-next-line df-rules/no-factgraph-save
        factGraph.save();
        for (const expectedFact of expectedCompleteFacts) {
          const finalHistory = history.concat({ route, setFacts: branch.setFacts });
          expect(
            factGraph.get(CPath.concretePath(expectedFact, collectionId)).complete,
            // This string interpolation is pretty nasty to read, but creates a pretty nice
            // message for the error. Hopefully more people will see/fix errors than will
            // need to modify this string.
            `Following the path and inputting facts: ${finalHistory
              .map((history) => {
                return `
    Route: ${history.route},
    ${history.setFacts.map((fact) => {
      return `FactPath: ${fact.path}, Value: ${
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        fact.value !== undefined ? (fact.value as any).toString() : `UNDEFINED`
      }`;
    })}`;
              })
              .join(``)}
left ${expectedFact} incomplete. Terminated on screen ${nextScreen}`
          ).toBe(true);
          sawOnlyKnockouts = false;
        }
      } else {
        // We have to remove the collection id for `getNextScreen` to work
        const nextRoute = nextScreen.routable.fullRoute(nextScreen.collectionId).split(`?`)[0];
        screenGraphsToTest.push({
          route: nextRoute,
          factGraphState: branch.factGraphJson,
          history: history.concat({ route, setFacts: branch.setFacts }),
        });
      }
    });
  }
  expect(
    sawOnlyKnockouts,
    `Starting at ${startingScreenRoute} lead to only knockout screens, never testing any fact completions`
  ).toBe(false);
}

function nextScreenIsTerminal(nextScreenUrl: string) {
  // eslint-disable-next-line eqeqeq
  return nextScreenUrl == `/checklist` || nextScreenIsKnockout(nextScreenUrl) || nextScreenUrl.startsWith(`/data-view`);
}

function nextScreenIsKnockout(nextScreenUrl: string) {
  return nextScreenUrl.includes(`knockout`);
}

function copyFactGraph(factGraph: FactGraph) {
  const data = factGraph.toJSON();
  return setupFactGraphDeprecated(JSON.parse(data)).factGraph;
}

interface FactGraphBranchState {
  factGraphJson: string;
  setFacts: { path: Path; value: unknown }[];
}

/**
 *
 * Returns a set of fact graphs that could exist to specify our next screen.
 * See getPossibleValuesForFact to see how we branch to create fact graphs.
 */
function getPotentialFactGraphsAfterScreen(
  screenRoute: string,
  factGraph: FactGraph,
  collectionId: string | null,
  factValuesToSet: Partial<Record<Path, unknown[]>>,
  leaveOptionalFieldsBlank: boolean
): FactGraphBranchState[] {
  const screenConfig = flow.screensByRoute.get(screenRoute);
  if (!screenConfig) {
    throw new Error(`No screen config for ${screenRoute}`);
  }
  // TODO: in the future, we need to include optional facts and branch on writing or *not* writing them
  const factConfigs = screenConfig.content.filter((c): c is FactConfig =>
    contentIsAvailableRequiredWritableFactConfig(c, factGraph, collectionId)
  );
  const factActions = screenConfig.setActions;
  const factActionSetFacts: { path: Path; value: unknown }[] = [];

  if (factActions.length > 0) {
    factActions.forEach((config) => {
      if (config.source === `df.language` || config.source === `emptyCollection`) return;
      const source = factGraph.get(CPath.concretePath(config.source, collectionId));
      if (source.hasValue) {
        const value = source.get;
        factGraph.set(CPath.concretePath(config.path, collectionId), value);
        factActionSetFacts.push({ path: config.path, value });
      }

      // eslint-disable-next-line df-rules/no-factgraph-save
      factGraph.save();
    });
  }
  // eslint-disable-next-line eqeqeq
  if (factConfigs.length == 0) {
    return [{ factGraphJson: factGraph.toJSON(), setFacts: factActionSetFacts }];
  }
  const possibleSettings = factConfigs.map((fc) => {
    return getPossibleValuesForFact(fc, factGraph, collectionId, factValuesToSet, leaveOptionalFieldsBlank);
  });
  const factsToSet = combinations(possibleSettings);
  const res: FactGraphBranchState[] = [];
  for (const factSet of factsToSet) {
    const tmpFactGraph = copyFactGraph(factGraph);
    for (const fact of factSet) {
      if (!fact.leaveEmpty) {
        tmpFactGraph.set(CPath.concretePath(fact.path, collectionId), fact.value);
      }
    }
    // eslint-disable-next-line df-rules/no-factgraph-save
    tmpFactGraph.save();
    res.push({ factGraphJson: tmpFactGraph.toJSON(), setFacts: [...factSet, ...factActionSetFacts] });
  }
  return res;
}

/**
 *
 * For a fact that we have on the screen, return an array of possible values
 * it might have.
 *
 * The function will return (in order of precedence):
 * 1. Any values for the path that are specified with factValuesToSet
 * 2. For an enum, all possible values
 * 3. For a boolean,true and false
 * 4. For other variables, a set of hard coded variables that may or may not make sense
 *    and you should consider overriding them with factValuesToSet for your test case
 *
 *
 * This function is still rather new, has not been used on all data types, and may have bugs.
 */
function getPossibleValuesForFact(
  fc: FactConfig,
  factGraph: FactGraph,
  collectionId: string | null,
  factValuesToSet: Partial<Record<Path, unknown[]>>,
  leaveOptionalFieldsBlank: boolean
): { path: Path; value: unknown; leaveEmpty?: true }[] {
  // eslint-disable-next-line eqeqeq
  if (leaveOptionalFieldsBlank && fc.props.required == false) {
    return [{ path: fc.props.path, value: undefined, leaveEmpty: true }];
  }
  const testValues = factValuesToSet[fc.props.path];
  if (testValues) {
    return testValues.map((value) => {
      return { path: fc.props.path as Path, value };
    });
  }
  const wrap = (value: unknown) => {
    return { path: fc.props.path, value };
  };
  if (fc.componentName === `Address`) {
    return [wrap(AddressFactory(`123 Fake St`, `Brend`, `10014`, `NY`, ``, `USA`).right)];
  } else if (fc.componentName === `Boolean`) {
    return [wrap(true), wrap(false)];
  } else if (fc.componentName === `Tin`) {
    return [wrap(TinFactory(`111111111`).right), wrap(TinFactory(`999999999`).right)];
  } else if (fc.componentName === `PhoneNumber`) {
    return [wrap(UsPhoneNumberFactory(`+14155552671`).right)];
  } else if (fc.componentName === `DatePicker`) {
    return [wrap(DayFactory(`1987-11-11`).right)];
  } else if (fc.componentName === `Ein`) {
    return [wrap(EinFactory(`111111111`).right)];
  } else if (fc.componentName === `Enum`) {
    const optionsPath = unwrapScalaOptional(factGraph.getDictionary().getOptionsPathForEnum(fc.props.path)) as Path;
    const result = factGraph.get(CPath.concretePath(optionsPath, collectionId));
    const values: string[] | undefined = result.complete
      ? scalaListToJsArray(result.get as ScalaList<string>)
      : undefined;
    if (!values) {
      throw new Error(`no enum values for ${optionsPath}`);
    }
    return values.map((v) => wrap(EnumFactory(v, optionsPath).right));
  } else if (fc.componentName === `MultiEnum`) {
    const optionsPath = unwrapScalaOptional(factGraph.getDictionary().getOptionsPathForEnum(fc.props.path)) as Path;
    const result = factGraph.get(CPath.concretePath(optionsPath, collectionId));
    const values: string[] | undefined = result.complete
      ? scalaListToJsArray(result.get as ScalaList<string>)
      : undefined;
    if (!values) {
      throw new Error(`no enum values for ${optionsPath}`);
    }
    return values.map((v) => wrap(MultiEnumFactory(v, optionsPath).right));
  } else if (fc.componentName === `GenericString`) {
    return [wrap(`GenericString`)];
  } else if (fc.componentName === `LimitingString`) {
    return [wrap(`LimitingString`)];
  } else if (fc.componentName === `Dollar`) {
    // maybe we should have more values
    return [wrap(DollarFactory(`1000`).right)];
  } else if (fc.componentName === `IpPin`) {
    return [wrap(IpPinFactory(`123456`).right)];
  } else if (fc.componentName === `FactSelect`) {
    throw new Error(`FactSelect not supported yet -- build it!`);
  } else if (fc.componentName === `BankAccount`) {
    return [wrap(BankAccountFactory(`Checking`, `021000021`, `123456`).right)];
  } else if (fc.componentName === `Pin`) {
    return [wrap(PinFactory(`12345`).right)];
  } else if (fc.componentName === `CollectionItemReference`) {
    throw new Error(`CollectionItemReference not supported yet -- build it!`);
  } else {
    assertNever(fc);
  }
}

// This is NP-hard! If you use it on a long, complicated section,
// your test will never terminate.
function combinations<T>(arr: T[][]): T[][] {
  if (arr.length === 0) return [[]];
  const res: T[][] = [];
  const [first, ...rest] = arr;
  const remaining = combinations(rest);
  first.forEach((e) => {
    remaining.forEach((smaller) => {
      res.push([e].concat(smaller));
    });
  });
  return res;
}
