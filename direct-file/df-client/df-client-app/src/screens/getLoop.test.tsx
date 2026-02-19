import { createFlowConfig } from '../flow/flowConfig.js';
import { FakeFactGraph } from '../misc/factgraphTestHelpers.js';
import { Flow, Category, Subcategory, SubSubcategory, Screen, CollectionLoop } from '../flow/flowDeclarations.js';
import { InfoDisplay, GenericString, Heading, SaveAndOrContinueButton } from '../flow/ContentDeclarations.js';
import { getFirstAvailableOfCollectionLoop } from '../misc/getCollectionLoopEnds.js';

const flow = (
  <Flow>
    <Category route='cat1'>
      <Subcategory route='sub1' completeIf={{ operator: `isComplete`, condition: `/filers/*/firstName` }}>
        <SubSubcategory route='subsub1'>
          <Screen route='one'>
            <Heading i18nKey='/foo' />
            <InfoDisplay i18nKey='/info/your-and-your-family/you/intro' />
            <SaveAndOrContinueButton />
          </Screen>
          <CollectionLoop loopName='firstUnconditional' collection='/filers'>
            <Screen route='firstUnconditional-correct'>
              <Heading i18nKey='/foo' />
              <GenericString path='/filers/*/firstName' />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='firstUnconditional-incorrect'>
              <Heading i18nKey='/foo' />
              <GenericString path='/filers/*/firstName' />
              <SaveAndOrContinueButton />
            </Screen>
          </CollectionLoop>

          <Screen route='two'>
            <Heading i18nKey='/foo' />
            <InfoDisplay i18nKey='/info/your-and-your-family/you/intro' />
            <SaveAndOrContinueButton />
          </Screen>
          <CollectionLoop loopName='firstTrue' collection='/filers'>
            <Screen route='firstTrue-correct' condition={{ operator: `isTrue`, condition: `/flowTrue` }}>
              <Heading i18nKey='/foo' />
              <GenericString path='/filers/*/firstName' />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='firstTrue-incorrect'>
              <Heading i18nKey='/foo' />
              <GenericString path='/filers/*/firstName' />
              <SaveAndOrContinueButton />
            </Screen>
          </CollectionLoop>

          <Screen route='three'>
            <Heading i18nKey='/foo' />
            <InfoDisplay i18nKey='/info/your-and-your-family/you/intro' />
            <SaveAndOrContinueButton />
          </Screen>
          <CollectionLoop loopName='firstFalse' collection='/filers'>
            <Screen route='firstFalse-incorrect' condition={{ operator: `isFalse`, condition: `/flowTrue` }}>
              <Heading i18nKey='/foo' />
              <GenericString path='/filers/*/firstName' />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='firstFalse-also-incorrect' condition={{ operator: `isFalse`, condition: `/flowTrue` }}>
              <Heading i18nKey='/foo' />
              <GenericString path='/filers/*/firstName' />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen route='firstFalse-correct' condition={{ operator: `isTrue`, condition: `/flowTrue` }}>
              <Heading i18nKey='/foo' />
              <GenericString path='/filers/*/firstName' />
              <SaveAndOrContinueButton />
            </Screen>
          </CollectionLoop>
        </SubSubcategory>
      </Subcategory>
    </Category>
  </Flow>
);

for (const scenario of [`firstUnconditional`, `firstTrue`, `firstFalse`]) {
  test(`identify the first available screen for loop ${scenario}`, () => {
    const flowConfig = createFlowConfig(flow);
    const fg = new FakeFactGraph(new Map([[`/flowTrue`, true]]));

    const nextScreenPath = getFirstAvailableOfCollectionLoop(scenario, fg, null, flowConfig);
    expect(nextScreenPath.routable.fullRoute(nextScreenPath.collectionId)).toBe(`/flow/cat1/sub1/${scenario}-correct`);
  });
}
