import { useCallback, useEffect, useMemo, useRef, useState, FC, useReducer, MutableRefObject } from 'react';
import { useFactGraph } from '../factgraph/FactGraphContext.js';
import { Path } from '../flow/Path.js';
import { ConcretePath } from '@irs/js-factgraph-scala';
import {
  ScreenContentConfig,
  contentConfigIsFactConfig,
  FactActionExternalSource,
  FlowComponentConfig,
} from '../flow/ContentDeclarations.js';
import { assertNever } from 'assert-never';
import { SetFactActionConfig } from '../flow/ScreenConfig.js';
import { useFilterContentContext } from '../context/FilterContentContext.js';
import { conditionsPass } from '../utils/condition.js';
import { useSaveAndPersist } from '../hooks/useSaveAndPersist.js';
import ScreenAlertAggregator from './ScreenAlertAggregator/ScreenAlertAggregator.js';
import SectionsAlertAggregator from './SectionsAlertAggregator/SectionsAlertAggregator.js';
import { extractUniqueAlertSummarySections } from '../misc/aggregatedAlertHelpers.js';
import { useChecklistState } from '../flow/useChecklistState.js';
import { AlertAggregatorType } from '../flow/flowDeclarations.js';
import { useTranslation } from 'react-i18next';
import { useIsReturnEditable } from '../hooks/useIsReturnEditable.js';
import {
  ScreenContentComponentName,
  HEADER_COMPONENT_NAMES,
  AGGREGATED_ALERT_COMPONENT_NAMES,
  AggregatedAlertComponentName,
  HeaderComponentName,
  buildRenderedScreenContentKey,
  useScreenInfo,
  setFactsFromFactActionPaths,
} from './screenUtils.js';
import { ScreenHeader } from './ScreenHeader.js';
import { RenderedScreenContent } from './RenderedScreenContent.js';
import { calculateScreenStatus } from '../flow/batches.js';
import styles from './Screen.module.scss';
export interface ScreenProps {
  screenRoute: string;
  screenContent: ScreenContentConfig[];
  collectionId: string | null;
  gotoNextScreen: () => void;
  setFactActionPaths: SetFactActionConfig[];
  alertAggregatorType?: AlertAggregatorType;
  componentsToHide?: ReadonlySet<ScreenContentConfig[`componentName`]>;
  collectionContext?: ConcretePath;
}

type ScreenSectionsAlertAggregatorProps = {
  factRefs: MutableRefObject<Map<ConcretePath, React.MutableRefObject<HTMLInputElement>>>;
};

type ComponentGroup<ComponentName extends ScreenContentComponentName> = {
  [Name in ComponentName]?: FlowComponentConfig<Name>[];
};
function groupConfigsByComponent<ComponentName extends ScreenContentComponentName>(
  configs: FlowComponentConfig<ComponentName>[]
): ComponentGroup<ComponentName> {
  const groupedConfigs = configs.reduce((group, config) => {
    const components = group.get(config.componentName) ?? [];

    components.push(config);

    group.set(config.componentName, components);

    return group;
  }, new Map<ScreenContentComponentName, FlowComponentConfig<ScreenContentComponentName>[]>());

  return Object.fromEntries(groupedConfigs) as ComponentGroup<ComponentName>;
}

function groupScreenContent(screenContent: ScreenContentConfig[]) {
  const headerConfigs = screenContent.filter((config): config is FlowComponentConfig<HeaderComponentName> =>
    (HEADER_COMPONENT_NAMES as ScreenContentComponentName[]).includes(config.componentName)
  );

  const aggregatedAlertConfigs = screenContent.filter(
    (config): config is FlowComponentConfig<AggregatedAlertComponentName> =>
      (AGGREGATED_ALERT_COMPONENT_NAMES as ScreenContentComponentName[]).includes(config.componentName)
  );

  const rest = screenContent.filter(
    (
      config
    ): config is Exclude<
      ScreenContentConfig,
      FlowComponentConfig<HeaderComponentName | AggregatedAlertComponentName>
    > =>
      !(
        (HEADER_COMPONENT_NAMES as ScreenContentComponentName[]).includes(config.componentName) ||
        (AGGREGATED_ALERT_COMPONENT_NAMES as ScreenContentComponentName[]).includes(config.componentName)
      )
  );

  return {
    headerConfigs,
    aggregatedAlertConfigs,
    rest,
  };
}

/**
 * We do not want to call useChecklistState for every Screen when only those using the sections style aggregator need it
 * Since  you cannot conditionally call hooks, but you can conditionally render components, this "component" is just a
 * wrapper for the SectionsAlertAggregator, which also calls useChecklistState if it is being rendered
 */
const ScreenSectionsAlertAggregator = ({ factRefs }: ScreenSectionsAlertAggregatorProps) => {
  const checklistState = useChecklistState();

  const { summaryErrorSections, summaryWarningSections } = extractUniqueAlertSummarySections(checklistState);

  return (
    <SectionsAlertAggregator
      refs={factRefs}
      summaryWarningSections={summaryWarningSections}
      summaryErrorSections={summaryErrorSections}
    />
  );
};

const Screen: FC<ScreenProps> = ({
  screenContent,
  gotoNextScreen,
  collectionId,
  screenRoute,
  setFactActionPaths,
  alertAggregatorType = `screen`,
  componentsToHide,
  collectionContext,
}) => {
  const { factGraph } = useFactGraph();
  const { shouldFilterContentBasedOnTaxState } = useFilterContentContext();
  const [, forceUpdate] = useReducer((x) => x + 1, 0);
  const { i18n } = useTranslation();
  const { isReturnEditable } = useIsReturnEditable();

  const screenInfo = useScreenInfo(screenRoute, screenContent, collectionId, componentsToHide);

  const initialValidity: [ConcretePath, boolean][] = screenInfo.visibleContent
    .filter(contentConfigIsFactConfig)
    .map((child) => [
      Path.concretePath(child.props.path, collectionId),

      // This line about CollectionItemReference is pretty gross, but they behave differently than
      // every other component in how they use placeholders, and if we don't include that behavior
      // then we can't make them actually required
      (child.componentName === `CollectionItemReference` &&
        factGraph.get(Path.concretePath(child.props.path, collectionId)).complete) ||
        (child.componentName !== `CollectionItemReference` &&
          factGraph.get(Path.concretePath(child.props.path, collectionId)).hasValue) ||
        // eslint-disable-next-line eqeqeq
        child.props.readOnly == true ||
        // eslint-disable-next-line eqeqeq
        child.props.required == false,
    ]);
  const [showFeedback, setShowFeedback] = useState(false);

  const [factValidity, setFactValidity] = useState(new Map<ConcretePath, boolean>(initialValidity));
  const factRefs = useRef(new Map<ConcretePath, React.MutableRefObject<HTMLInputElement>>());

  const setPathFactValidity = useCallback(
    (path: ConcretePath, validity: boolean) => {
      setFactValidity((fv) => {
        if (fv.get(path) === validity) return fv;
        else return new Map(fv).set(path, validity);
      });
    },
    [setFactValidity]
  );

  const saveAndPersist = useSaveAndPersist();

  useEffect(() => {
    // This if statement ensures that we only save if we've modified values
    const setFacts = async () => {
      if (setFactActionPaths.length > 0 && isReturnEditable) {
        setFactsFromFactActionPaths(factGraph, collectionId, setFactActionPaths);
        // This doesn't need runIfUpdatesArePossible because we don't even try
        // to process fact actions if the return isn't editable
        await saveAndPersist();
        // Force render to ensure facts display.
        forceUpdate();
      }
    };
    setFacts();
  }, [setFactActionPaths, collectionId, factGraph, saveAndPersist, forceUpdate, isReturnEditable]);

  // Set language facts separately to avoid unnecessarily invoking other `SetFactAction`s when the language changes
  // TODO: Move this into a separate hook when the Screen is refactored
  useEffect(() => {
    const setFacts = async () => {
      let isDirty = false;
      if (setFactActionPaths.length > 0 && isReturnEditable) {
        setFactActionPaths
          .filter(
            (action): action is { path: SetFactActionConfig[`path`]; source: FactActionExternalSource } =>
              action.source.startsWith(`df`) && conditionsPass(action, factGraph, collectionId)
          )
          .forEach(({ path, source }) => {
            if (source === `df.language`) {
              const targetPath = Path.concretePath(path, collectionId);
              factGraph.set(targetPath, i18n.resolvedLanguage);
              isDirty = true;
            } else if (source === `emptyCollection`) {
              // do nothing, we should never actually get here since we filter above
            } else {
              assertNever(source);
            }
          });
      }
      // This doesn't need runIfUpdatesArePossible because we don't even try
      // to process fact actions if the return isn't editable
      if (isDirty) await saveAndPersist();
    };

    setFacts();
  }, [setFactActionPaths, collectionId, factGraph, saveAndPersist, i18n.resolvedLanguage, isReturnEditable]);

  const showSummaryErrorAlert = Array.from(factValidity.values()).some((value) => value === false);

  const { headings, icons, contextHeadings, mefAlerts, taxReturnAlerts, filteredContent } = useMemo(() => {
    const { headerConfigs, aggregatedAlertConfigs, rest } = groupScreenContent(screenInfo.visibleContent);

    // TODO: Is there a better way to do this?
    const { Heading, ContextHeading, IconDisplay } = groupConfigsByComponent(headerConfigs);
    if (shouldFilterContentBasedOnTaxState) {
      if (Heading?.length !== 1) throw new Error(`Page ${screenRoute} did not have exactly one Heading config`);
      if (ContextHeading && ContextHeading.length > 1)
        throw new Error(`Page ${screenRoute} had more than one ContextHeading config`);
      if (IconDisplay && IconDisplay.length > 1)
        throw new Error(`Page ${screenRoute} had more than one IconDisplay config`);
    }

    const mefAlerts = aggregatedAlertConfigs.filter(
      (config): config is FlowComponentConfig<`MefAlert`> => config.componentName === `MefAlert`
    );
    const taxReturnAlerts = aggregatedAlertConfigs.filter(
      (config): config is FlowComponentConfig<`TaxReturnAlert`> => config.componentName === `TaxReturnAlert`
    );

    return {
      headings: Heading ?? [],
      icons: IconDisplay ?? [],
      contextHeadings: ContextHeading ?? [],
      mefAlerts,
      taxReturnAlerts,
      filteredContent: rest,
    };
  }, [screenInfo.visibleContent, screenRoute, shouldFilterContentBasedOnTaxState]);

  // TODO: MeF Alerts go with the rest of the screen warnings.
  const formGroupCount = document.getElementsByClassName(`usa-form-group`).length;
  const headingIsDraft = headings.some(
    (h) =>
      calculateScreenStatus(h.props.batches).isOpen ||
      contextHeadings.some((ch) => calculateScreenStatus(ch.props.batches).isOpen)
  );

  return (
    <>
      {alertAggregatorType === `sections` ? (
        <ScreenSectionsAlertAggregator factRefs={factRefs} />
      ) : (
        <ScreenAlertAggregator
          collectionId={collectionId}
          factRefs={factRefs}
          factValidity={factValidity}
          mefAlerts={mefAlerts}
          taxReturnAlerts={taxReturnAlerts}
          showSummary={showFeedback && formGroupCount > 1 && showSummaryErrorAlert}
          screenInfo={screenInfo}
          showFeedback={showFeedback}
        />
      )}

      <ScreenHeader
        isDraft={headingIsDraft}
        headings={headings}
        contextHeadings={contextHeadings}
        icons={icons}
        collectionId={collectionId}
      />

      <div className='usa-prose'>
        <form id='the_form' className='input-group vertical'>
          {filteredContent.map((config) => {
            const key = buildRenderedScreenContentKey(config);
            const content = (
              <RenderedScreenContent
                key={key}
                config={config}
                gotoNextScreen={gotoNextScreen}
                showFeedback={showFeedback}
                forceUpdate={forceUpdate}
                setShowFeedback={setShowFeedback}
                setPathFactValidity={setPathFactValidity}
                collectionId={collectionId}
                factValidity={factValidity}
                factRefs={factRefs}
                screenInfo={screenInfo}
                collectionContext={collectionContext}
              />
            );
            const batches = config.props.batches || [];
            const isOpen = calculateScreenStatus(batches).isOpen;
            if (isOpen) {
              return (
                <div key={key} className={styles.draftContent}>
                  {content}
                </div>
              );
            }
            return content;
          })}
        </form>
      </div>
    </>
  );
};

export default Screen;
