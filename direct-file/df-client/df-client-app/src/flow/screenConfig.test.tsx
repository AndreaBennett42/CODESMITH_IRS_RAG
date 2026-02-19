import {
  Enum,
  ScreenContentNode,
  SetFactAction,
  InfoDisplay,
  GenericString,
  CollectionItemManager,
} from './ContentDeclarations.js';
import { beforeEach, it, describe, expect } from 'vitest';
import { buildFactActionFromConfigNode, buildScreenContentFromConfigNode } from './ScreenConfig.js';

describe(`ScreenContentConfigParser`, () => {
  interface LocalTestContext {
    genericString: ScreenContentNode;
    enumor: ScreenContentNode;
    collectionItemManager: ScreenContentNode;
    infoDisplay: ScreenContentNode;
    setFact: ScreenContentNode;
  }

  beforeEach<LocalTestContext>(async (context) => {
    context.infoDisplay = <InfoDisplay i18nKey='cats' />;
    context.genericString = <GenericString path='/address' />;
    context.enumor = <Enum path='/maritalStatus' />;
    context.collectionItemManager = <CollectionItemManager path='/formW2s' loopName='w2s' donePath='/formW2sIsDone' />;
    context.setFact = <SetFactAction path='/interestIncome' source='/ordinaryDividends' />;
  });
  it<LocalTestContext>(`grabs the node type`, ({ infoDisplay, enumor, genericString, collectionItemManager }) => {
    expect(buildScreenContentFromConfigNode(infoDisplay)?.componentName).toBe(`InfoDisplay`);
    expect(buildScreenContentFromConfigNode(enumor)?.componentName).toBe(`Enum`);
    expect(buildScreenContentFromConfigNode(genericString)?.componentName).toBe(`GenericString`);
    expect(buildScreenContentFromConfigNode(collectionItemManager)?.componentName).toBe(`CollectionItemManager`);
  });

  it<LocalTestContext>(`returns undefined if you pass it a node that doesn't map to ScreenContentConfig`, ({
    setFact,
  }) => {
    const field = buildScreenContentFromConfigNode(setFact);
    expect(field).toBeUndefined();
  });
});

describe(`FactActionParser`, () => {
  interface LocalTestContext {
    genericString: ScreenContentNode;
    enumor: ScreenContentNode;
    collectionItemManager: ScreenContentNode;
    infoDisplay: ScreenContentNode;
    setFact: ScreenContentNode;
  }

  beforeEach<LocalTestContext>(async (context) => {
    context.infoDisplay = <InfoDisplay i18nKey='cats' />;
    context.genericString = <GenericString path='/address' />;
    context.enumor = <Enum path='/maritalStatus' />;
    context.collectionItemManager = <CollectionItemManager path='/formW2s' loopName='w2s' donePath='/formW2sIsDone' />;
    context.setFact = <SetFactAction path='/interestIncome' source='/ordinaryDividends' />;
  });
  it<LocalTestContext>(`returns undefined for ScreenContentConfig nodes`, ({
    infoDisplay,
    genericString,
    enumor,
    collectionItemManager,
  }) => {
    expect(buildFactActionFromConfigNode(infoDisplay)).toBeUndefined();
    expect(buildFactActionFromConfigNode(genericString)).toBeUndefined();
    expect(buildFactActionFromConfigNode(enumor)).toBeUndefined();
    expect(buildFactActionFromConfigNode(collectionItemManager)).toBeUndefined();
  });
  it<LocalTestContext>(`returns a SetFactActionConfig for a SetFactAction`, ({ setFact }) => {
    const setFactConfig = buildFactActionFromConfigNode(setFact);
    expect(setFactConfig).toStrictEqual({
      path: `/interestIncome`,
      source: `/ordinaryDividends`,
      condition: undefined,
      conditions: undefined,
    });
  });
});
