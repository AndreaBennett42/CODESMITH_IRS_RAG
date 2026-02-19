/* eslint-disable max-len */
import { getTaxReturnLoopAlerts } from './aggregatedAlertHelpers.js';
import { FlowCollectionLoop } from '../flow/flowConfig.js';
import { createBooleanWrapper } from '../test/persistenceWrappers.js';
import { setupFactGraphDeprecated } from '../test/setupFactGraph.js';

const uuid1 = `959c03d1-af4a-447f-96aa-d19397048a44`;
const uuid2 = `888c03d1-af4a-447f-96aa-d19397048a44`;

const loops = [
  {
    loopName: `/familyAndHousehold`,
    autoIterate: false,
    subcategoryRoute: `/flow/you-and-your-family/dependents`,
    iconName: `Person`,
    collectionName: `/familyAndHousehold`,
    donePath: `/familyAndHouseholdIsDone`,
    dataViewSections: [],
    collectionItemCompletedCondition: `/familyAndHousehold/*/isCompleted`,
    screens: [
      {
        route: `qualified-dependent-ip-pin-ready`,
        fullRoute: () => undefined,
        subSubcategoryRoute: `/flow/you-and-your-family/dependents/claim-choice`,
        subcategoryRoute: `/flow/you-and-your-family/dependents`,
        categoryRoute: `/flow/you-and-your-family`,
        screenRoute: `/flow/you-and-your-family/dependents/qualified-dependent-ip-pin-ready`,
        conditions: [`/flowTrue`],
        collectionContext: `/familyAndHousehold`,
        collectionLoop: {
          loopName: `/familyAndHousehold`,
          autoIterate: false,
        },
        routeAutomatically: true,
        actAsDataView: false,
        content: [
          {
            componentName: `TaxReturnAlert`,
            props: {
              i18nKey: `/info/you-and-your-family/dependents/ip-pin-not-ready`,
              conditions: [`/familyAndHousehold/*/tpClaims`],
              internalLink: `/flow/you-and-your-family/dependents/qualified-dependent-ip-pin-ready`,
              type: `warning`,
              headingLevel: `h2`,
              accordionHeadingLevel: `h3`,
            },
          },
          {
            componentName: `TaxReturnAlert`,
            props: {
              i18nKey: `/info/you-and-your-family/dependents/ip-pin-not-ready`,
              conditions: [`/familyAndHousehold/*/isUsCitizenFullYear`],
              internalLink: `/flow/you-and-your-family/dependents/qualified-dependent-ip-pin-ready`,
              type: `warning`,
              headingLevel: `h2`,
              accordionHeadingLevel: `h3`,
            },
          },
          {
            componentName: `TaxReturnAlert`,
            props: {
              i18nKey: `/info/you-and-your-family/dependents/ip-pin-not-ready`,
              conditions: [`/familyAndHousehold/*/isUsCitizenFullYear`],
              internalLink: `/flow/you-and-your-family/dependents/qualified-dependent-ip-pin-ready`,
              type: `error`,
              headingLevel: `h2`,
              accordionHeadingLevel: `h3`,
            },
          },
        ],
        _factPaths: [`/familyAndHousehold/*/flowIpPinReady`],
        _setActions: [],
      },
    ],
  },
] as unknown as FlowCollectionLoop[];

describe(getTaxReturnLoopAlerts.name, () => {
  test(`gets all alerts in loops`, () => {
    const { factGraph } = setupFactGraphDeprecated({
      '/familyAndHousehold': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [uuid1, uuid2] },
      },
      [`/familyAndHousehold/#${uuid1}/tpClaims`]: createBooleanWrapper(true),
      [`/familyAndHousehold/#${uuid1}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      [`/familyAndHousehold/#${uuid2}/isUsCitizenFullYear`]: createBooleanWrapper(true),
    });
    const result = getTaxReturnLoopAlerts(true, loops, factGraph);
    expect(result.errors.length).toBe(1);
    expect(result.warnings.length).toBe(2);
  });
});
