import { FC, forwardRef } from 'react';
import styles from './AllScreens.module.scss';
import { GatedTreeNode, useFlow } from '../flow/flowConfig.js';
import { useTranslation } from 'react-i18next';
import { Path } from '../fact-dictionary/Path.js';
import { ScreenConfig } from '../flow/ScreenConfig.js';
import { AllScreensCsv } from './AllScreensCsv.js';
import { AllScreensPostSubmission } from './postSubmission/AllScreensPostSubmission.js';
import { AllScreensScreenWrapper } from './AllScreensScreenWrapper.js';
import { dependenciesPerTaxTest, uuid } from './AllScreensContext.js';
import { Condition, RawCondition, rawConditionToString } from '../flow/Condition.js';
import { facts } from '../fact-dictionary/generated/facts.js';
import { unmapFactAlias } from '../fact-dictionary/generate-src/dependencyGraph.js';
import { RawFact } from '../fact-dictionary/FactTypes.js';
import { Icon, Tooltip } from '@trussworks/react-uswds';
import { BATCH_NAME, BatchStates, BatchStatus, screenIsInStatus, screenIsInWorkflowStep } from '../flow/batches.js';
import { AllScreensDataview } from './AllScreensDataview.js';

const factsByPath = (() => {
  const ret: { [key: string]: RawFact } = {};
  for (const fact of facts) {
    ret[fact[`@path`]] = fact;
  }
  return ret;
})();

const WORD_WRAP_MAX = 80;

export const showTypes = [`screens`, `csv`, `postSubmission`] as const;
export type ShowType = (typeof showTypes)[number];

export interface AllScreensFilterSet {
  tax: Path | undefined;
  batch: BATCH_NAME | undefined;
  status: BatchStates | undefined;
  workflow: BatchStatus | undefined;
}

/**
 *
 * @returns This route exists as a secret feature for the team to view all of the screens at once!
 */
type AllScreensContentProps = {
  showType: ShowType;
  showConditionalScreenLogic: boolean;
  hideAlerts: boolean;
  filters: AllScreensFilterSet;
  showModals: boolean;
  showDataviews: boolean;
};

const AllScreensContent: FC<AllScreensContentProps> = ({
  showType,
  showConditionalScreenLogic,
  hideAlerts,
  filters,
  showModals,
  showDataviews,
}) => {
  const flow = useFlow();
  const { t } = useTranslation();

  const filterScreens = (screen: ScreenConfig) => {
    // By default, include the screen. Then run through our filters and let them knock the
    // screen out if a filter is set and the screen doesn't match.
    let includeScreen = true;

    // Tax Test filter
    if (filters.tax) {
      const dependenciesForTaxTest = filters.tax ? dependenciesPerTaxTest[filters.tax] : undefined;
      if (dependenciesForTaxTest) {
        includeScreen = screen.factPaths.some((f) => dependenciesForTaxTest.has(f));
      }
    }

    // Batch filter
    if (filters.batch) {
      includeScreen = screen.batches.includes(filters.batch);
    }

    // Screen Status filter
    if (filters.status) {
      includeScreen = screenIsInStatus(filters.status, screen.batches);
    }

    // Workflow Step filter
    if (filters.workflow) {
      includeScreen = screenIsInWorkflowStep(filters.workflow, screen.batches);
    }

    return includeScreen;
  };

  return (
    <main className={styles.mainContent}>
      {showType === `csv` && <AllScreensCsv />}
      {showType === `screens` &&
        flow.categories.map((cat) => {
          const heading = t(`checklist.${cat.route}.heading`);
          const headingId = heading.replace(/\s+/g, `-`).toLowerCase();
          return (
            <div key={cat.route}>
              <h2 id={headingId}>{heading}</h2>
              {cat.subcategories.map((subcat) => {
                const heading = t(`checklist.${subcat.route}.heading`);
                const headingId = heading.replace(/\s+/g, `-`).toLowerCase();
                return (
                  <div key={subcat.route}>
                    <h3 id={headingId}>{heading}</h3>
                    <div key={subcat.route} className={styles.subcontentContainer} tabIndex={0}>
                      {showConditionalScreenLogic &&
                        subcat.treeNodes.map((node) => renderFlowTreeNode(node, filterScreens, hideAlerts, showModals))}
                      {!showConditionalScreenLogic &&
                        subcat.screens
                          .filter(filterScreens)
                          .map((screen) => (
                            <AllScreensScreenWrapper
                              key={screen.fullRoute(uuid)}
                              screen={screen}
                              hideAlerts={hideAlerts}
                              showModals={showModals}
                            />
                          ))}
                    </div>
                    {showDataviews && <AllScreensDataview t={t} subcat={subcat} heading={heading} />}
                  </div>
                );
              })}
            </div>
          );
        })}
      {showType === `postSubmission` && <AllScreensPostSubmission />}
    </main>
  );
};

function renderFlowTreeNode(
  node: GatedTreeNode | ScreenConfig,
  filterScreen: (sc: ScreenConfig) => boolean,
  hideAlerts: boolean,
  showModals: boolean
) {
  if (flowNodeScreenIsScreen(node)) {
    if (filterScreen(node)) {
      return (
        <AllScreensScreenWrapper
          key={node.fullRoute(uuid)}
          screen={node}
          hideAlerts={hideAlerts}
          showModals={showModals}
        />
      );
    } else {
      return null;
    }
  }

  const allScreensInNode: ScreenConfig[] = [];
  const queue = [...node.screens];
  while (queue.length > 0) {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const screen = queue.pop()!;
    if (flowNodeScreenIsScreen(screen)) {
      allScreensInNode.push(screen);
    } else {
      queue.push(...screen.screens);
    }
  }
  const filteredScreens = allScreensInNode.filter(filterScreen);

  // eslint-disable-next-line eqeqeq
  if (filteredScreens.length == 0) {
    return null;
  } else {
    const content = (
      <div style={{ display: `flex` }}>
        {node.screens.map((sc) => renderFlowTreeNode(sc, filterScreen, hideAlerts, showModals))}
      </div>
    );
    if (node.gates.length > 0) {
      return (
        <div className={styles.gateWrapper}>
          <div className={styles.gateName}>
            {node.gates.map((condition) => (
              <ConditionView condition={condition} key={rawConditionToString(condition)} />
            ))}
          </div>
          {content}
        </div>
      );
    } else {
      return content;
    }
  }
}

function flowNodeScreenIsScreen(scr: ScreenConfig | GatedTreeNode): scr is ScreenConfig {
  // eslint-disable-next-line eqeqeq
  return (scr as ScreenConfig).content != undefined;
}

const ConditionView = ({ condition }: { condition: RawCondition }) => {
  const rawPath = new Condition(condition).innerCondition.factPath;
  const path = rawPath ? unmapFactAlias(rawPath) : undefined;
  const fact = path ? factsByPath[path] : undefined;
  const factDescription = fact?.Description;
  const CustomInfoButton = forwardRef(CustomInfoButtonForwardRef);
  const factRender = factDescription ? (
    <Tooltip<{ children: React.ReactNode }>
      label={wordWrap(factDescription)}
      position='bottom'
      asCustom={CustomInfoButton}
    >
      <Icon.Help aria-hidden='true' />
    </Tooltip>
  ) : null;
  return (
    <span>
      {rawConditionToString(condition)} {factRender}
    </span>
  );
};

type CustomInfoButtonProps = React.PropsWithChildren<unknown> &
  JSX.IntrinsicElements['span'] &
  React.RefAttributes<HTMLSpanElement>;

const CustomInfoButtonForwardRef: React.ForwardRefRenderFunction<HTMLSpanElement, CustomInfoButtonProps> = (
  { children, ...tooltipProps }: CustomInfoButtonProps,
  ref
) => (
  <span ref={ref} {...tooltipProps}>
    {children}
  </span>
);

/**
 * This method takes a string, and adds newlines at a certain character count
 * We use this to take the ugly descriptions from our fact dictionary XML
 * and make sure that it is always pretty wrapped.
 */
const wordWrap = (str: string) =>
  str
    .replaceAll(`\n`, ` `) // remove existing newlines
    .replaceAll(/\s+/g, ` `) // remove strange whitespaces
    .replace(new RegExp(`(?![^\\n]{1,${WORD_WRAP_MAX}}$)([^\\n]{1,${WORD_WRAP_MAX}})\\s`, `g`), `$1\n`); // and wrap

export default AllScreensContent;
