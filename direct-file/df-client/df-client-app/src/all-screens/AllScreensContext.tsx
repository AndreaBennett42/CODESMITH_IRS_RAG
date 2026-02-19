import { FC, ReactNode } from 'react';
import { HelmetProvider } from 'react-helmet-async';
import { StaticRouter } from 'react-router-dom/server.js';
import { FactGraphTranslationContext } from '../context/FactGraphTranslationContext.js';
import { FilterContentContext } from '../context/FilterContentContext.js';
import { SystemAlertContext, getEmptySystemAlertsMap } from '../context/SystemAlertContext/SystemAlertContext.js';
import { TaxReturnsContext } from '../context/TaxReturnsContext.js';
import { NetworkConnectionContext } from '../context/networkConnectionContext.js';
import { Path } from '../fact-dictionary/Path.js';
import { generateDependencyGraph, getDeepDependencies } from '../fact-dictionary/generate-src/dependencyGraph.js';
import { FactGraphContext } from '../factgraph/FactGraphContext.js';
import { createBooleanWrapper } from '../test/persistenceWrappers.js';
import { AllScreensStateTaxesSettingsContextProvider } from './postSubmission/AllScreensPostSubmissionSettings.js';
import { wrappedFacts } from '../fact-dictionary/generated/wrappedFacts.js';
import { InterceptingFactGraph } from '../factgraph/InterceptingFactGraph.js';
import { Provider } from 'react-redux';
import { store } from '../redux/store.js';

export const TAX_TESTS: Path[] = [
  `/eligibleForHoh`,
  `/eligibleForQss`,
  `/eligibleForMFS`,
  `/eligibleForMFJ`,
  `/eligibleForSingle`,
  `/eitcQualified`,
  `/ctcQualified`,
  `/odcQualified`,
  `/treatFilersAsDependents`,
  `/filersCouldHaveDependentOrQualifyingPerson`,
  `/familyAndHousehold/*/qualifyingChild`,
  `/familyAndHousehold/*/qualifyingRelative`,
  `/familyAndHousehold/*/eligibleDependent`,
  `/filersCouldQualifyForEitc`,
  `/filerCouldQualifyForHoh`,
  `/familyAndHousehold/*/eligibleForBenefitSplit`,
  `/familyAndHousehold/*/eligibleCtc`,
  `/familyAndHousehold/*/eligibleOdc`,
  `/isPaperPath`,
];

export const uuid = `959c03d1-af4a-447f-96aa-d19397048a44`;

const collections = wrappedFacts.filter((f) => f.writable?.typeName === `Collection`).map((f) => f.path);

export const AllScreensContext: FC<{ children: ReactNode }> = ({ children }) => {
  const collectionInit = Object.fromEntries(
    // Every writable collection is initialized from the start.
    collections.map((c) => [c, { $type: `gov.irs.factgraph.persisters.CollectionWrapper`, item: { items: [uuid] } }])
  );

  const factGraph = new InterceptingFactGraph({
    ...collectionInit,
    [`/filers/#${uuid}/isPrimaryFiler`]: createBooleanWrapper(true),
    [`/hohQualifyingPerson`]: {
      $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
      item: { id: uuid },
    },
  });

  return (
    <AllScreensStateTaxesSettingsContextProvider>
      <NetworkConnectionContext.Provider value={{ online: true, prevOnlineStatus: true }}>
        <SystemAlertContext.Provider
          value={{
            systemAlerts: getEmptySystemAlertsMap(),
            setSystemAlert: () => undefined,
            deleteSystemAlert: () => undefined,
          }}
        >
          <FactGraphTranslationContext.Provider value={{ shouldFetchTranslationValuesFromFactGraph: false }}>
            <FilterContentContext.Provider value={{ shouldFilterContentBasedOnTaxState: false }}>
              <TaxReturnsContext.Provider
                value={{
                  currentTaxReturnId: null,
                  taxReturns: [],
                  fetchTaxReturns: () => undefined,
                  isFetching: false,
                  fetchSuccess: false,
                }}
              >
                <FactGraphContext.Provider value={{ factGraph }}>
                  <HelmetProvider>
                    <Provider store={store}>
                      <StaticRouter location='/'>{children}</StaticRouter>
                    </Provider>
                  </HelmetProvider>
                </FactGraphContext.Provider>
              </TaxReturnsContext.Provider>
            </FilterContentContext.Provider>
          </FactGraphTranslationContext.Provider>
        </SystemAlertContext.Provider>
      </NetworkConnectionContext.Provider>
    </AllScreensStateTaxesSettingsContextProvider>
  );
};

export const dependenciesPerTaxTest = (() => {
  const dependencyMap = generateDependencyGraph();
  const ret: { [key: string]: Set<Path> } = {};
  for (const test of TAX_TESTS) {
    ret[test] = getDeepDependencies(test, dependencyMap);
  }
  return ret;
})();
