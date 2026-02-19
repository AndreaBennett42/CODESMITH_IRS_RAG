import { ConcretePath } from '@irs/js-factgraph-scala';
import { assertNever } from 'assert-never';
import { Dispatch, SetStateAction, DispatchWithoutAction, createElement, useContext } from 'react';
import {
  ScreenContentConfig,
  FlowComponentConfig,
  contentConfigIsFactConfig,
  contentConfigIsCollectionItemManagerConfig,
  contentConfigIsInfoDisplayConfig,
  contentConfigIsInternalLinkConfig,
  contentConfigIsCollectionDataViewInternalLinkConfig,
  contentConfigIsDFAlertConfig,
  contentConfigIsConditionalListConfig,
  contentConfigIsScreenButtonConfig,
  contentConfigIsDataPreviewConfig,
  contentConfigIsCollectionDataPreviewConfig,
} from '../flow/ContentDeclarations.js';
import { useSaveAndPersistIfPossible } from '../hooks/useSaveAndPersistIfPossible.js';
import { InfoDisplayProps, ScreenButtonProps } from '../types/core.js';
import {
  FactTypeRenderer,
  CollectionItemManagerRenderer,
  InfoTypeRenderer,
  InternalLinkRenderer,
  CollectionDataViewInternalLinkRenderer,
  DFAlertRenderer,
  ConditionalListRenderer,
  ScreenButtonRenderer,
  DataPreviewRenderer,
  CollectionDataPreviewRenderer,
} from './factTypes/index.js';
import {
  HeaderComponentName,
  AggregatedAlertComponentName,
  buildRenderedScreenContentKey,
  focusOnErrorOrSummary,
  useGetFactRef,
  ScreenInfo,
} from './screenUtils.js';
import { Path } from '../flow/Path.js';
import { useIsReturnEditable } from '../hooks/useIsReturnEditable.js';
import { conditionsPass } from '../utils/condition.js';
import { useFactGraph } from '../factgraph/FactGraphContext.js';
import { TaxReturnsContext } from '../context/TaxReturnsContext.js';

export const RenderedScreenContent = ({
  config,
  collectionId,
  showFeedback,
  setPathFactValidity,
  factValidity,
  setShowFeedback,
  factRefs,
  gotoNextScreen,
  forceUpdate,
  screenInfo,
  collectionContext,
}: {
  config: Exclude<ScreenContentConfig, FlowComponentConfig<HeaderComponentName | AggregatedAlertComponentName>>;
  collectionId: string | null;
  factValidity: Map<ConcretePath, boolean>;
  factRefs: React.MutableRefObject<Map<ConcretePath, React.MutableRefObject<HTMLInputElement>>>;
  screenInfo: ScreenInfo;
  // NOTE: These are marked optional to make this component more convenient outside Screen.tsx
  showFeedback?: boolean;
  setPathFactValidity?: (path: ConcretePath, validity: boolean) => void;
  setShowFeedback?: Dispatch<SetStateAction<boolean>>;
  gotoNextScreen?: () => void;
  forceUpdate?: DispatchWithoutAction;
  collectionContext?: ConcretePath;
}) => {
  const saveAndPersistIfPossible = useSaveAndPersistIfPossible();
  const { currentTaxReturnId } = useContext(TaxReturnsContext);
  const isReturnEditable = useIsReturnEditable();
  const handleFactRef = useGetFactRef(factRefs);
  const { factGraph } = useFactGraph();

  const children =
    `childConfigs` in config.props
      ? config.props.childConfigs
          .filter(
            (config) =>
              // eslint-disable-next-line eqeqeq
              conditionsPass(config.props, factGraph, collectionId) && config.props.displayOnlyOn != `data-view`
          )
          .map((config) => (
            <RenderedScreenContent
              key={buildRenderedScreenContentKey(config)}
              config={config}
              collectionId={collectionId}
              showFeedback={showFeedback}
              setPathFactValidity={setPathFactValidity}
              factValidity={factValidity}
              setShowFeedback={setShowFeedback}
              factRefs={factRefs}
              gotoNextScreen={gotoNextScreen}
              forceUpdate={forceUpdate}
              screenInfo={screenInfo}
              collectionContext={collectionContext}
            />
          ))
      : [];

  // We've pulled the header and the icon out of the content array, because we want it to appear above the
  // error screens, whereas this content is below the error screen.
  if (contentConfigIsFactConfig(config) || contentConfigIsCollectionItemManagerConfig(config)) {
    const concretePath = Path.concretePath(config.props.path, collectionId);

    if (showFeedback === undefined) {
      throw new Error(`Tried to render a Fact control, but showFeedback was undefined`);
    }
    if (setPathFactValidity === undefined) {
      throw new Error(`Tried to render a Fact control, but setPathFactValidity was undefined`);
    }

    const baseProps = {
      concretePath: concretePath,
      showFeedback,
      collectionId,
      onValidData: setPathFactValidity,
      isValid: factValidity.get(concretePath) || false,
      factRefs: factRefs,
      handleFactRef,
      factValidity,
    };

    // TODO: I wish we had a better way of mapping these
    if (contentConfigIsFactConfig(config)) {
      const readOnly = config.props.readOnly ?? !isReturnEditable;
      const renderer = FactTypeRenderer[config.componentName];
      const renderedFact = createElement(renderer, {
        readOnly,
        ...config.props,
        ...baseProps,
        required: config.props.required !== false,
        collectionId,
        ref: handleFactRef(concretePath),
        saveAndPersist: saveAndPersistIfPossible,
      });

      return <div>{renderedFact}</div>;
    } else if (contentConfigIsCollectionItemManagerConfig(config)) {
      const renderer = CollectionItemManagerRenderer[config.componentName];
      const loopName = config.props.loopName;
      const loop = config.props.loop;
      const baseProps = { path: config.props.path, concretePath, donePath: config.props.donePath };
      const shouldSeeHubCompletionBtnsPath = config.props.shouldSeeHubCompletionBtnsPath;
      return createElement(renderer, {
        ...baseProps,
        loop,
        loopName,
        shouldSeeHubCompletionBtnsPath,
        saveAndPersist: saveAndPersistIfPossible,
      });
    } else {
      return assertNever(config);
    }
  } else if (contentConfigIsInfoDisplayConfig(config)) {
    // TODO: Fix typing problems with InfoTypeRenderer components
    const renderer = InfoTypeRenderer[config.componentName] as React.FunctionComponent<InfoDisplayProps>;
    return createElement(renderer, {
      ...(config.props as InfoDisplayProps),
      collectionId,
      showFeedback,
      gotoNextScreen,
      taxId: currentTaxReturnId || undefined,
    });
  } else if (contentConfigIsInternalLinkConfig(config)) {
    const renderer = InternalLinkRenderer[config.componentName];
    return createElement(renderer, {
      ...config.props,
      collectionId,
    });
  } else if (contentConfigIsCollectionDataViewInternalLinkConfig(config)) {
    const renderer = CollectionDataViewInternalLinkRenderer[config.componentName];
    return createElement(renderer, {
      ...config.props,
    });
  } else if (contentConfigIsDFAlertConfig(config)) {
    // TaxReturnAlerts and MefAlerts are rendered using the alert aggregation system.
    // Standard DFAlerts are enabled as alert-style, informational components which render in the Screen body.
    const renderer = DFAlertRenderer[config.componentName];

    const { childConfigs: _, ...props } = {
      ...config.props,
      collectionId,
      internalLink: config.props.internalLink || screenInfo.screenRoute,
    };

    return createElement(renderer, props, children);
  } else if (contentConfigIsConditionalListConfig(config)) {
    const renderer = ConditionalListRenderer[config.componentName];
    return createElement(renderer, {
      ...config.props,
      collectionId,
    });
  } else if (contentConfigIsScreenButtonConfig(config)) {
    const renderer = ScreenButtonRenderer[config.componentName] as React.FunctionComponent<ScreenButtonProps>;

    if (!gotoNextScreen) throw new Error(`Tried to render ScreenButton but goToNextScreen was not available `);
    if (!setShowFeedback) throw new Error(`Tried to render ScreenButton but setShowFeedback was not available `);

    return createElement(renderer, {
      ...config.props,
      collectionId,
      showFeedback,
      gotoNextScreen,
      taxId: currentTaxReturnId || undefined,
      screenHasFacts: screenInfo.hasFacts,
      factValidity,
      setShowFeedback,
      focusOnErrorOrSummary,
      collectionContext,
    });
  } else if (contentConfigIsDataPreviewConfig(config)) {
    const renderer = DataPreviewRenderer[config.componentName];

    return createElement(renderer, {
      ...config.props,
      currentRoute: screenInfo.screenRoute,
      collectionId: collectionId,
    });
  } else if (contentConfigIsCollectionDataPreviewConfig(config)) {
    const renderer = CollectionDataPreviewRenderer[config.componentName];
    if (!collectionContext) throw new Error(`Tried to render CollectionDataPreview, but no collection context`);
    return createElement(renderer, {
      ...config.props,
      collectionContext,
    });
  } else {
    return assertNever(config);
  }
};
