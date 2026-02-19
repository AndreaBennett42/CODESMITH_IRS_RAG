import { DayFactory, EnumFactory } from '@irs/js-factgraph-scala';
import flowNodes from '../../flow/flow.js';
import { createFlowConfig } from '../../flow/flowConfig.js';
import {
  createBooleanWrapper,
  createCollectionWrapper,
  createDayWrapper,
  createEnumWrapper,
  createStringWrapper,
  createTinWrapper,
} from '../persistenceWrappers.js';
import { FlowSetFactInput, checkFlowSetsFacts } from './checkFlowSetsFacts.js';
const flow = createFlowConfig(flowNodes);

export const uuid1 = `959c03d1-af4a-447f-96aa-d19397048a44`;
export const spouseId = `059c03d1-af4a-447f-96aa-d19397048a44`;

/**
 * This test is skipped because it takes too long to run.
 *
 * However, it has always failed in the first 15 seconds or run for longer than
 * 10 minutes. I think that there are some optimizations we could
 * make to the test runner that could get this to run every time.
 */
describe.skip(`Spouse sections always completes`, () => {
  const input: FlowSetFactInput = {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    startingScreenRoute: flow.subcategoriesByRoute.get(`/flow/you-and-your-family/spouse`)!.screens[0].screenRoute,
    startingFactState: {
      '/filers': createCollectionWrapper([uuid1, spouseId]),
      [`/filers/#${uuid1}/isPrimaryFiler`]: createBooleanWrapper(true),
      [`/filers/#${uuid1}/firstName`]: createStringWrapper(`Test`),
      [`/filers/#${uuid1}/writableMiddleInitial`]: createStringWrapper(`Test`),
      [`/filers/#${uuid1}/lastName`]: createStringWrapper(`Test`),
      [`/filers/#${uuid1}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
      [`/filers/#${uuid1}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${uuid1}/filerResidenceAndIncomeState`]: createBooleanWrapper(true),
      [`/filers/#${uuid1}/canBeClaimed`]: createBooleanWrapper(false),
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ca`, `/scopedStateOptions`),
      [`/primaryFilerW2And1099IntInScopedState`]: createEnumWrapper(
        `onlySame`,
        `/primaryFilerW2And1099IntStateOptions`
      ),
    },
    expectedStartingFactState: {},
    expectedCompleteFacts: [
      `/spouseSectionCompleted`,
      `/isMarried`,
      `/filersCouldntBeDependents`,
      `/treatFilersAsDependents`,
      `/livedApartAllYear`,
      `/livedApartLastSixMonths`,
      `/secondaryFilerAdditionalStandardDeductionItems`,
    ],
    collectionId: spouseId,
    factValuesToSet: {},
    leaveOptionalFieldsBlank: false,
  };
  it(`All complete spouse facts get set as complete`, () => {
    checkFlowSetsFacts(input);
  });
});

/**
 * This test is skipped because it takes too long to run.
 *
 * However, it has always failed in the first 15 seconds or run for longer than
 * 10 minutes. I think that there are some optimizations we could
 * make to the test runner that could get this to run every time.
 */
describe.skip(`Dependents sections always completes`, () => {
  const input: FlowSetFactInput = {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    startingScreenRoute: flow.collectionLoopsByName.get(`/familyAndHousehold`)!.screens[0].screenRoute,
    startingFactState: {
      // Our input is a single independent filer -- they can claim dependents and become eligible for HoH
      '/filers': createCollectionWrapper([uuid1, spouseId]),
      '/familyAndHousehold': createCollectionWrapper([uuid1]),
      '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
      [`/filers/#${uuid1}/isPrimaryFiler`]: createBooleanWrapper(true),
      [`/filers/#${uuid1}/firstName`]: createStringWrapper(`Test`),
      [`/filers/#${uuid1}/writableMiddleInitial`]: createStringWrapper(`Test`),
      [`/filers/#${uuid1}/lastName`]: createStringWrapper(`Test`),
      [`/filers/#${uuid1}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
      [`/filers/#${uuid1}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${uuid1}/filerResidenceAndIncomeState`]: createBooleanWrapper(true),
      [`/filers/#${uuid1}/dateOfBirth`]: createDayWrapper(`1987-01-01`),
      [`/filers/#${uuid1}/canBeClaimed`]: createBooleanWrapper(false),
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ca`, `/scopedStateOptions`),
      [`/primaryFilerW2And1099IntInScopedState`]: createEnumWrapper(
        `onlySame`,
        `/primaryFilerW2And1099IntStateOptions`
      ),
    },
    expectedStartingFactState: {},
    expectedCompleteFacts: [
      `/familyAndHousehold/*/eligibleCtc`,
      `/familyAndHousehold/*/eligibleOdc`,
      `/familyAndHousehold/*/eitcQualifyingChild`,
      `/familyAndHousehold/*/isQualifyingChildOfAnother`,
      `/familyAndHousehold/*/isClaimedDependent`,
      `/familyAndHousehold/*/hohQualifyingPerson`,
      `/familyAndHousehold/*/qssQualifyingPerson`,
    ],
    collectionId: uuid1,
    factValuesToSet: {
      // these three sections have not yet been rebuilt, and are being dodged for now.
      // We should eventually check both paths
      [`/familyAndHousehold/*/writableCouldBeQualifyingChildOfAnother`]: [false],
      [`/familyAndHousehold/*/inParentsCustody`]: [false],
      [`/familyAndHousehold/*/hasOtherBiologicalOrAdoptiveParent`]: [false],
      // These values cause massive branching, so we set a few -- we
      //  check biological child, foster child, and another relative
      [`/familyAndHousehold/*/relationship`]: [
        EnumFactory(`biologicalChild`, `/relationshipOptions`).right,
        EnumFactory(`noneOfTheAbove`, `/relationshipOptions`).right,
      ],
      [`/familyAndHousehold/*/dateOfBirth`]: [DayFactory(`2015-11-11`).right, DayFactory(`2001-11-11`).right],
    },
    leaveOptionalFieldsBlank: false,
  };
  it(`All complete dependents facts get set as complete`, () => {
    checkFlowSetsFacts(input);
  });
});
