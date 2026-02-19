import styles from './AllScreens.module.scss';
import { DataViewFact, SubSubCategoryHeader } from '../screens/data-view/SubSubCategory.js';
import {
  contentConfigIsFactConfig,
  contentConfigIsInfoDisplayConfig,
  InfoDisplayConfig,
  ScreenContentConfig,
} from '../flow/ContentDeclarations.js';
import dataviewStyles from '../screens/data-view/SubSubCategory.module.scss';
import { FlowAssertion, FlowSubcategory } from '../flow/flowConfig.js';
import { TFunction } from 'i18next';
import { rawConditionToString } from '../flow/Condition.js';
import { uuid } from './AllScreensContext.js';
import Assertion from '../screens/data-view/Assertion.js';
import { AlertConfigs, MefAlertConfig } from '../misc/aggregatedAlertHelpers.js';
import { AbsolutePath } from '../fact-dictionary/Path.js';

const getUniqueKey = (factConfig: ScreenContentConfig, screenRoute: string) => {
  const path = `path` in factConfig.props && factConfig.props.path;
  let uniqueKey = `${screenRoute}-${path}`;
  if (`conditions` in factConfig.props && factConfig.props.conditions !== undefined) {
    const add = factConfig.props.conditions.map(rawConditionToString).join(`\n`);
    uniqueKey = `${uniqueKey}-${add}`;
  } else if (`condition` in factConfig.props && factConfig.props.condition !== undefined) {
    const add = rawConditionToString(factConfig.props.condition);
    uniqueKey = `${uniqueKey}-${add}`;
  }
  return uniqueKey;
};

export const AllScreensDataview = ({
  subcat,
  t,
  heading,
}: {
  subcat: FlowSubcategory;
  t: TFunction;
  heading: string;
}) => {
  let sscs = subcat.subSubcategories;
  // eslint-disable-next-line eqeqeq
  if (sscs.length == 0) {
    sscs = subcat.loops.flatMap((loop) => loop.subSubcategories);
  }
  const assertionsByPath = (() => {
    const ret: { [key: string]: FlowAssertion[] } = {};
    for (const assertion of subcat.assertions) {
      const key = assertion.subSubCategoryRoute || assertion.subcategoryRoute;
      if (key in ret) {
        ret[key].push(assertion);
      } else {
        ret[key] = [assertion];
      }
    }
    return ret;
  })();

  const isContextHeading = (content: ScreenContentConfig) => {
    const isContextHeading = contentConfigIsInfoDisplayConfig(content) && content.componentName === `ContextHeading`;
    return isContextHeading && content.props.displayOnlyOn !== `edit`;
  };

  return (
    <div className={styles.dataviewContainer}>
      <h3>{`${heading} dataview`}</h3>
      <div key={subcat.route} className={styles.subcontentContainer} tabIndex={0}>
        {sscs.map((ssc) => {
          const screens = ssc.screens.flatMap((screen) => screen);
          const screenAssertions = ssc.screens
            .flatMap((sc) =>
              sc.content.filter((c) => c.componentName === `FactAssertion` || c.componentName === `FactResultAssertion`)
            )
            .filter((result) => result);
          const sscAssertions = assertionsByPath[ssc.fullRoute] || [];
          // eslint-disable-next-line eqeqeq
          if (screenAssertions.length == 0 && ssc.screens.length == 0) {
            return null;
          }
          const emptyMefAlerts: AlertConfigs<MefAlertConfig> = {
            errors: [],
            warnings: [],
          };
          const emptyTaxReturnAlerts: AlertConfigs = {
            errors: [],
            warnings: [],
          };

          const contextHeading =
            ssc.screens.length > 0
              ? ssc.screens[0].content.find((content): content is InfoDisplayConfig => isContextHeading(content))
              : null;
          // Get all facts on screens
          const facts = screens.flatMap((screen) => {
            const factsOnScreen = screen.content.flatMap((factConfig) => {
              const isFact = contentConfigIsFactConfig(factConfig);
              const isOptional = isFact ? factConfig.props.required === false : false;
              if (!isFact) {
                return [];
              }

              const path = (`path` in factConfig.props && factConfig.props.path) || (`` as AbsolutePath);
              // eslint-disable-next-line eqeqeq
              if ((isFact || isOptional) && factConfig.props.displayOnlyOn != `edit`) {
                return {
                  path,
                  value: getUniqueKey(factConfig, screen.route), // hack! storing a unique key for display purposes
                  type: ``,
                  isSensitive: `isSensitive` in factConfig.props && factConfig.props.isSensitive,
                  collectionId: uuid,
                  editRoute: (`editRoute` in factConfig.props && factConfig.props.editRoute) || ``,
                  isCheckbox: (`inputType` in factConfig.props && factConfig.props.inputType === `checkbox`) || false,
                  anchorLink: (`dataViewAnchorLink` in factConfig.props && factConfig.props.dataViewAnchorLink) || ``,
                  isReadOnly: (`readOnly` in factConfig.props && factConfig.props.readOnly) || false,
                  isNextIncompleteFact: false,
                  mefAlerts: emptyMefAlerts,
                  taxReturnAlerts: emptyTaxReturnAlerts,
                };
              }
              return [];
            });
            return factsOnScreen;
          });

          return (
            <div className={styles.screenOuterContainer} key={ssc.fullRoute}>
              <div className={styles.screenContainer} key={ssc.fullRoute} style={{ display: `block` }}>
                <div className={styles.dataviewHeader}>
                  <h3>{ssc.routeSuffix}</h3>
                </div>
                <div
                  data-testid='subsubcategory'
                  key={ssc.fullRoute}
                  className={`${dataviewStyles.dataViewSection} dataview-section`}
                >
                  <SubSubCategoryHeader
                    ssc={ssc}
                    i18nKey={`subsubcategories.${ssc.subcategoryRoute}.${ssc.routeSuffix}`}
                    collectionId={null}
                    action={`edit`}
                    headingContext={{}}
                    headingLevel={ssc.headingLevel}
                    contextHeading={contextHeading || undefined}
                  />
                  {screenAssertions.map((assertion) => {
                    return (
                      <Assertion
                        key={`${assertion.props.i18nKey}-${uuid}`}
                        i18nKey={assertion.props.i18nKey}
                        type={`type` in assertion.props ? assertion.props.type : `info`}
                        editRoute={`editRoute` in assertion.props ? assertion.props.editRoute : ``}
                        collectionId={null}
                      />
                    );
                  })}
                  {sscAssertions.map((assertion) => {
                    return (
                      <Assertion
                        key={`${assertion.i18nKey}-${uuid}`}
                        i18nKey={assertion.i18nKey}
                        type={assertion.type}
                        editRoute={assertion.editRoute}
                        collectionId={uuid}
                      />
                    );
                  })}
                  <ul className={`${dataviewStyles.dataViewItems}`}>
                    {facts.map((fact) => {
                      return (
                        <DataViewFact
                          key={fact.value}
                          fact={fact}
                          ssc={ssc}
                          collectionId={uuid}
                          sectionIsComplete={true}
                          t={t}
                        />
                      );
                    })}
                  </ul>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default AllScreensDataview;
