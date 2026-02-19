import getNextScreen from './getNextScreen.js';
import { FakeFactGraph } from '../misc/factgraphTestHelpers.js';
import { Flow, Gate, Category, Subcategory, SubSubcategory, Screen, CollectionLoop } from '../flow/flowDeclarations.js';
import {
  InfoDisplay,
  GenericString,
  Heading,
  SaveAndOrContinueButton,
  CollectionItemManager,
} from '../flow/ContentDeclarations.js';
import { ConditionString, RawCondition } from '../flow/Condition.js';
import { createFlowConfig } from '../flow/flowConfig.js';
import { AbsolutePath } from '../fact-dictionary/Path.js';

const flow = (
  <Flow>
    <Category route='cat'>
      <Subcategory route='sub' completeIf={{ operator: `isComplete`, condition: `/filers/*/lastName` }}>
        <SubSubcategory route='subsub1'>
          <Screen route='one'>
            <Heading i18nKey='/foo' />
            <InfoDisplay i18nKey='/info/your-and-your-family/you/intro' />
            <SaveAndOrContinueButton />
          </Screen>

          <Screen route='two'>
            <Heading i18nKey='/foo' />
            <GenericString path='/filers/*/firstName' />
            <GenericString path='/filers/*/lastName' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='three' condition={`/isTrue` as RawCondition}>
            <Heading i18nKey='/foo' />
            <GenericString path='/address' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='four'>
            <Heading i18nKey='/foo' />
            <GenericString path={`/invalid` as AbsolutePath} />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='five' routeAutomatically={false}>
            <Heading i18nKey='/foo' />
            <GenericString path={`/invalid` as AbsolutePath} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </Subcategory>
      <Subcategory route='sub2' completeIf={{ operator: `isComplete`, condition: `/somePath3` as ConditionString }}>
        <SubSubcategory route='subsub2'>
          <Screen route='one'>
            <Heading i18nKey='/foo' />
            <GenericString path={`/somePath1` as AbsolutePath} />
            <SaveAndOrContinueButton />
          </Screen>
          <Gate condition={`/topCondition` as RawCondition}>
            <Screen route='two'>
              <Heading i18nKey='/foo' />
              <GenericString path={`/somePath2` as AbsolutePath} />
              <SaveAndOrContinueButton />
            </Screen>
            <Gate condition={`/nestedCondition` as RawCondition}>
              <Screen route='three'>
                <Heading i18nKey='/foo' />
                <GenericString path={`/somePath3` as AbsolutePath} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='four' condition={`/localCondition` as RawCondition}>
                <Heading i18nKey='/foo' />
                <GenericString path={`/somePath4` as AbsolutePath} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='five'>
                <Heading i18nKey='/foo' />
                <GenericString path={`/somePath5` as AbsolutePath} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='six'>
                <Heading i18nKey='/foo' />
                <GenericString path={`/somePath6` as AbsolutePath} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='seven'>
                <Heading i18nKey='/foo' />
                <GenericString path={`/somePath7` as AbsolutePath} />
                <SaveAndOrContinueButton />
              </Screen>
            </Gate>
            <Screen route='eight'>
              <Heading i18nKey='/foo' />
              <GenericString path={`/somePath8` as AbsolutePath} />
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>
          <Screen route='nine' actAsDataView={true}>
            <Heading i18nKey='/foo' />
            <GenericString path={`/somePath9` as AbsolutePath} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </Subcategory>
      <Subcategory route='sub3' completeIf={{ operator: `isComplete`, condition: `/somePath3` as ConditionString }}>
        <SubSubcategory route='subsub3'>
          <Screen route='one'>
            <Heading i18nKey='/foo' />
            <GenericString path={`/somePath1` as AbsolutePath} />
            <SaveAndOrContinueButton />
          </Screen>
          <Gate condition={`/topCondition` as RawCondition}>
            <Screen route='two'>
              <Heading i18nKey='/foo' />
              <GenericString path={`/somePath2` as AbsolutePath} />
              <SaveAndOrContinueButton />
            </Screen>
            <Gate condition={`/nestedCondition` as RawCondition}>
              <Screen route='three'>
                <Heading i18nKey='/foo' />
                <GenericString path={`/somePath3` as AbsolutePath} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='four' condition={`/localCondition` as RawCondition}>
                <Heading i18nKey='/foo' />
                <GenericString path={`/somePath4` as AbsolutePath} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='five'>
                <Heading i18nKey='/foo' />
                <GenericString path={`/somePath5` as AbsolutePath} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='six'>
                <Heading i18nKey='/foo' />
                <GenericString path={`/somePath6` as AbsolutePath} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='seven'>
                <Heading i18nKey='/foo' />
                <GenericString path={`/somePath7` as AbsolutePath} />
                <SaveAndOrContinueButton />
              </Screen>
            </Gate>
            <Screen route='eight' actAsDataView={true}>
              <Heading i18nKey='/foo' />
              <GenericString path={`/somePath8` as AbsolutePath} />
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>
          <Screen route='nine'>
            <Heading i18nKey='/foo' />
            <GenericString path={`/somePath9` as AbsolutePath} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </Subcategory>
      <Subcategory
        route='sub4'
        completeIf={{ operator: `isComplete`, condition: `/filers/*/lastName` }}
        skipDataView={true}
      >
        <SubSubcategory route='subsub4'>
          <Screen route='one'>
            <Heading i18nKey='/foo' />
            <InfoDisplay i18nKey='/info/your-and-your-family/you/intro' />
            <SaveAndOrContinueButton />
          </Screen>
          <Screen route='two'>
            <Heading i18nKey='/foo' />
            <GenericString path='/filers/*/firstName' />
            <GenericString path='/filers/*/lastName' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </Subcategory>
      <Subcategory
        route='sub5'
        completeIf={{ operator: `isComplete`, condition: `/filers/*/lastName` }}
        collectionContext='/interestReports'
      >
        <Screen route='one'>
          <Heading i18nKey='/foo' />
          <InfoDisplay i18nKey='/info/your-and-your-family/you/intro' />
          <SaveAndOrContinueButton />
        </Screen>
        <CollectionLoop
          loopName='/interestReports'
          collection='/interestReports'
          iconName='AttachMoney'
          collectionItemCompletedCondition='/interestReports/*/isComplete'
          donePath='/interestReportsIsDone'
        >
          <Screen route='two'>
            <Heading
              i18nKey='/heading/income/interest/add'
              condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
            />
            <SaveAndOrContinueButton />
          </Screen>
        </CollectionLoop>
        <Screen route='three'>
          <Heading i18nKey='/heading/income/interest/int-summary' />
          <SaveAndOrContinueButton />
        </Screen>
      </Subcategory>
      <Subcategory route='sub6' completeIf={{ operator: `isComplete`, condition: `/filers/*/lastName` }}>
        <Screen route='one'>
          <Heading i18nKey='/foo' />
          <InfoDisplay i18nKey='/info/your-and-your-family/you/intro' />
          <SaveAndOrContinueButton />
        </Screen>
      </Subcategory>
      <Subcategory route='sub7' completeIf={{ operator: `isComplete`, condition: `/sub6Done` as ConditionString }}>
        {/* Subcategory that contains an autoiterating and inner loop */}
        <Screen route='one'>
          <Heading i18nKey='/foo' />
          <InfoDisplay i18nKey='/info/your-and-your-family/you/intro' />
          <SaveAndOrContinueButton />
        </Screen>
        <SubSubcategory route='ssc1'>
          <Screen route='two'>
            <Heading i18nKey='/foo' />
            <GenericString path={`/somePath2` as AbsolutePath} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='sscInner' completeIf='/cdccCareProvidersIsDone' collectionContext='/cdccCareProviders'>
          <Screen route='three'>
            <Heading i18nKey='/foo' />
            <CollectionItemManager
              path='/cdccCareProviders'
              loopName='/cdccCareProviders'
              donePath='/cdccCareProvidersIsDone'
            />
          </Screen>
          <CollectionLoop
            loopName='/cdccCareProviders'
            collection='/cdccCareProviders'
            collectionItemCompletedCondition='/cdccCareProviders/*/isComplete'
            donePath='/cdccCareProvidersIsDone'
            isInner={true}
            dataViewSections={[
              {
                i18nKey: `dataviews./flow/credits-and-deductions/credits/care-providers.your-care-providers`,
                condition: `/flowTrue`,
              },
            ]}
          >
            <SubSubcategory route='loopcat1'>
              <Screen route='four'>
                <Heading i18nKey='/foo' />
                <GenericString path={`/somePath4` as AbsolutePath} />
                <SaveAndOrContinueButton />
              </Screen>
            </SubSubcategory>
          </CollectionLoop>
        </SubSubcategory>
        <SubSubcategory route='ssc3'>
          <Screen route='five'>
            <Heading i18nKey='/foo' />
            <GenericString path={`/somePath5` as AbsolutePath} />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
      </Subcategory>
    </Category>
  </Flow>
);

test(`getNextScreen returns the next screen if it doesn't have a condition`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(new Map([[`/isTrue`, true]]));
  const currentScreen = flowConfig.screensByRoute.get(`/flow/cat/sub/one`) ?? fail();
  const expectedNextScreen = flowConfig.screensByRoute.get(`/flow/cat/sub/two`) ?? fail();
  const nextScreenPath = getNextScreen(currentScreen, fg, null, flowConfig);
  expect(nextScreenPath.routable).toBe(expectedNextScreen);
});

test(`getNextScreen returns the next screen if it has a condition that is true`, () => {
  const flowConfig = createFlowConfig(flow);

  const fg = new FakeFactGraph(new Map([[`/isTrue`, true]]));
  const currentScreen = flowConfig.screensByRoute.get(`/flow/cat/sub/two`) ?? fail();
  const nextScreenPath = getNextScreen(currentScreen, fg, null, flowConfig);
  const expectedNextScreen = flowConfig.screensByRoute.get(`/flow/cat/sub/three`) ?? fail();
  expect(nextScreenPath.routable).toBe(expectedNextScreen);
});

test(`getNextScreen skips the next screen if it has a condition that is false`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(new Map([[`/isTrue`, false]]));
  const currentScreen = flowConfig.screensByRoute.get(`/flow/cat/sub/two`) ?? fail();
  const nextScreenPath = getNextScreen(currentScreen, fg, null, flowConfig);
  const expectedNextScreen = flowConfig.screensByRoute.get(`/flow/cat/sub/four`) ?? fail();
  expect(nextScreenPath.routable).toBe(expectedNextScreen);
});

test(`getNextScreen skips screens that do not route automatically`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(new Map());
  const currentScreen = flowConfig.screensByRoute.get(`/flow/cat/sub/four`) ?? fail();
  const nextScreenPath = getNextScreen(currentScreen, fg, null, flowConfig);
  expect(nextScreenPath.routable.fullRoute(null)).toBe(`/data-view/flow/cat/sub`);
});

test(`getNextScreen goes to the checklist at the end of a section if no data view`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(new Map());
  const currentScreen = flowConfig.screensByRoute.get(`/flow/cat/sub4/two`) ?? fail();
  const nextScreenPath = getNextScreen(currentScreen, fg, null, flowConfig);
  expect(nextScreenPath.routable.fullRoute(null)).toBe(`/checklist`);
});

test(`getNextScreen goes to the data view at the end of a section`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(new Map());
  const currentScreen = flowConfig.screensByRoute.get(`/flow/cat/sub/five`) ?? fail();
  const nextScreenPath = getNextScreen(currentScreen, fg, null, flowConfig);
  expect(nextScreenPath.routable.fullRoute(null)).toBe(`/data-view/flow/cat/sub`);
});

test(`getNextScreen goes to the checklist if the screen acting as a data view is last`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(new Map());
  const currentScreen = flowConfig.screensByRoute.get(`/flow/cat/sub2/nine`) ?? fail();
  const nextScreenPath = getNextScreen(currentScreen, fg, null, flowConfig);
  expect(nextScreenPath.routable.fullRoute(null)).toBe(`/checklist`);
});

test(`getNextScreen goes to the checklist if the screen comes after a collection loop`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(new Map());
  const currentScreen = flowConfig.screensByRoute.get(`/flow/cat/sub5/three`) ?? fail();
  const nextScreenPath = getNextScreen(currentScreen, fg, null, flowConfig);
  expect(nextScreenPath.routable.fullRoute(null)).toBe(`/checklist`);
});

test(`getNextScreen goes to the next screen after the screen acting as a data view`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(new Map());
  const currentScreen = flowConfig.screensByRoute.get(`/flow/cat/sub3/eight`) ?? fail();
  const nextScreenPath = getNextScreen(currentScreen, fg, null, flowConfig);
  expect(nextScreenPath.routable.fullRoute(null)).toBe(`/flow/cat/sub3/nine`);
});

test(`getNextScreen goes to the first screen of an inner collection loop`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(new Map());
  const currentScreen = flowConfig.screensByRoute.get(`/flow/cat/sub7/two`) ?? fail();
  const nextScreenPath = getNextScreen(currentScreen, fg, null, flowConfig);
  expect(nextScreenPath.routable.fullRoute(null)).toBe(`/flow/cat/sub7/three`);
});

test(`getNextScreen goes from last screen of inner loop to first screen of outer loop`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(new Map());
  const currentScreen = flowConfig.screensByRoute.get(`/flow/cat/sub7/four`) ?? fail();
  const nextScreenPath = getNextScreen(currentScreen, fg, null, flowConfig);
  expect(nextScreenPath.routable.fullRoute(null)).toBe(`/flow/cat/sub7/five`);
});

test(`getNextScreen calculates nested conditions: all true`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(
    new Map([
      [`/topCondition`, true],
      [`/nestedCondition`, true],
      [`/localCondition`, true],
    ])
  );
  const currentScreen = flowConfig.screensByRoute.get(`/flow/cat/sub2/four`) ?? fail();
  const expectedNextScreen = flowConfig.screensByRoute.get(`/flow/cat/sub2/five`) ?? fail();
  const nextScreenPath = getNextScreen(currentScreen, fg, null, flowConfig);
  expect(nextScreenPath.routable).toBe(expectedNextScreen);
});

test(`getNextScreen calculates nested conditions:  top false`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(
    new Map([
      [`/topCondition`, false],
      [`/nestedCondition`, true],
      [`/localCondition`, true],
    ])
  );
  const screenOne = flowConfig.screensByRoute.get(`/flow/cat/sub2/one`) ?? fail();
  const screenThree = flowConfig.screensByRoute.get(`/flow/cat/sub2/three`) ?? fail();
  expect(getNextScreen(screenOne, fg, null, flowConfig).routable.fullRoute(null)).toBe(`/flow/cat/sub2/nine`);
  expect(getNextScreen(screenThree, fg, null, flowConfig).routable.fullRoute(null)).toBe(`/flow/cat/sub2/nine`);
});

test(`getNextScreen calculates nested conditions: nested false`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(
    new Map([
      [`/topCondition`, true],
      [`/nestedCondition`, false],
      [`/localCondition`, true],
    ])
  );
  const screenTwo = flowConfig.screensByRoute.get(`/flow/cat/sub2/two`) ?? fail();
  const screenThree = flowConfig.screensByRoute.get(`/flow/cat/sub2/three`) ?? fail();
  expect(getNextScreen(screenTwo, fg, null, flowConfig).routable.fullRoute(null)).toBe(`/flow/cat/sub2/eight`);
  expect(getNextScreen(screenThree, fg, null, flowConfig).routable.fullRoute(null)).toBe(`/flow/cat/sub2/eight`);
});

test(`getNextScreen calculates nested conditions: inner false`, () => {
  const flowConfig = createFlowConfig(flow);
  const fg = new FakeFactGraph(
    new Map([
      [`/topCondition`, true],
      [`/nestedCondition`, true],
      [`/localCondition`, false],
    ])
  );
  const screenThree = flowConfig.screensByRoute.get(`/flow/cat/sub2/three`) ?? fail();
  const expectedNextScreen = flowConfig.screensByRoute.get(`/flow/cat/sub2/five`) ?? fail();
  expect(getNextScreen(screenThree, fg, null, flowConfig).routable).toBe(expectedNextScreen);
});
