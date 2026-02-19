import { getFirstAvailableOfCollectionLoop, getEndOfCollectionLoop } from '../../../misc/getCollectionLoopEnds.js';
import getNextScreen from '../../../screens/getNextScreen.js';
import { NavigateFunction, To, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import Translation from '../../../components/Translation/index.js';
import { Button, Icon } from '@trussworks/react-uswds';
import { useFactGraph } from '../../../factgraph/FactGraphContext.js';
import { CollectionFactory, ConcretePath, scalaListToJsArray } from '@irs/js-factgraph-scala';
import { useCallback, FC, useMemo } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Path } from '../../../flow/Path.js';
import { useFlow, FlowCollectionLoop } from '../../../flow/flowConfig.js';
import { Condition } from '../../../flow/Condition.js';
import { useKnockoutCheck } from '../../../hooks/useKnockoutCheck.js';
import { AbsolutePath, Path as FGPath, WritableAbsolutePath } from '../../../fact-dictionary/Path.js';
import { useIsReturnEditable } from '../../../hooks/useIsReturnEditable.js';
import { TFunction } from 'i18next';

type ButtonUpOneLevelProps = {
  loop?: FlowCollectionLoop;
  navigate: NavigateFunction;
  knockoutRoute?: string;
  t: TFunction;
};

const ButtonsUp = ({ loop, navigate, knockoutRoute, t }: ButtonUpOneLevelProps) => {
  const handleButtonUp = () => {
    if (knockoutRoute) {
      navigate(knockoutRoute);
    } else {
      const whereTo = `/data-view${loop?.subcategoryRoute}`;
      if (whereTo) {
        navigate(whereTo);
      }
    }
  };
  const handleButtonChecklist = () => {
    if (knockoutRoute) {
      navigate(knockoutRoute);
    } else {
      navigate(`/checklist`);
    }
  };

  return (
    <>
      {loop?.isInner && (
        <div>
          <Button type='button' onClick={handleButtonUp}>
            <Translation
              i18nKey={`fields.generics.collectionLoops.gotoPrefix`}
              collectionId={null}
              context={{ destination: t(`checklist.${loop?.subcategoryRoute}.heading`) }}
            />
            <Icon.NavigateNext size={3} className='usa-button__icon-right' aria-hidden='true' />
          </Button>
        </div>
      )}
      <Button type='button' outline={loop?.isInner} onClick={handleButtonChecklist}>
        <Translation i18nKey={`fields.generics.collectionLoops.gotoChecklist`} collectionId={null} />
        {loop?.isInner && <Icon.NavigateFarNext size={3} className='usa-button__icon-right' aria-hidden='true' />}
        {!loop?.isInner && <Icon.NavigateNext size={3} className='usa-button__icon-right' aria-hidden='true' />}
      </Button>
    </>
  );
};

export interface CollectionItemControlProps {
  path: FGPath;
  concretePath: ConcretePath;
  shouldSeeHubCompletionBtnsPath?: AbsolutePath;
  donePath?: WritableAbsolutePath;
  loopName: string;
  loop?: FlowCollectionLoop;
  knockoutRoute?: string;
}

export const CollectionItemManager: FC<
  CollectionItemControlProps & {
    saveAndPersist: () => Promise<{ hasPersistError: boolean }>;
  }
> = ({ path, loopName, concretePath, donePath, loop, saveAndPersist, knockoutRoute }) => {
  const { t } = useTranslation(`translation`);
  const flow = useFlow();
  const { factGraph } = useFactGraph();

  const navigate = useNavigate();
  const result = factGraph.get(concretePath);
  const { getIsKnockedOut } = useKnockoutCheck();
  const { isReturnEditable } = useIsReturnEditable();

  const collectionItems: string[] = useMemo(() => {
    return result.complete ? scalaListToJsArray(result.get.getItemsAsStrings()) : [];
  }, [result]);

  const getNextScreenInFlow = () => {
    const endOfLoop = getEndOfCollectionLoop(loopName, factGraph, null, flow, false);
    if (endOfLoop !== undefined) {
      // TODO -- I think there is a bug here if we have two back to back loops
      // without a different manager in between, but right now we never do that.
      const whereTo = getNextScreen(endOfLoop, factGraph, null, flow);
      if (whereTo.routable === null) {
        throw new Error(`Collection Loop ${loopName} doesn't have an end --- is there a typo?`);
      }
      return whereTo;
    }
    return null;
  };

  const complete = async () => {
    let hasPersistError = false;
    let hasChangeToPersist = false;

    if (!factGraph.get(concretePath).complete) {
      factGraph.set(concretePath, CollectionFactory([]));
      hasChangeToPersist = true;
    }
    if (donePath !== undefined) {
      factGraph.set(Path.concretePath(donePath, null), true);
      hasChangeToPersist = true;
    }
    if (hasChangeToPersist) {
      const persistResult = await saveAndPersist();
      hasPersistError = persistResult.hasPersistError;
    }

    // It's possible that, after saving the fact graph above, the filer is knocked out.
    // In the income section, we have a couple knockouts that only fire once the filer
    // has said they're done entering a certain type of form (e.g., social security income)
    // So, if they are, we're not going to find an end of the collection loop and need to
    // instead send them to the knockout
    if (!hasPersistError && knockoutRoute !== undefined && getIsKnockedOut()) {
      navigate(knockoutRoute);
    }
    const whereTo = getNextScreenInFlow();
    if (!hasPersistError && whereTo) {
      navigate(whereTo.routable.fullRoute(whereTo.collectionId));
    }
  };

  const goToKnockoutOrNavigate = (to: To) => {
    if (knockoutRoute !== undefined && getIsKnockedOut()) {
      navigate(knockoutRoute);
    } else {
      navigate(to);
    }
  };

  const create = useCallback(async () => {
    const newId = uuidv4();
    const newCollectionItems = [...collectionItems, newId];
    factGraph.set(concretePath, CollectionFactory(newCollectionItems));
    let hasPersistError = false;
    const persistResult = await saveAndPersist();
    hasPersistError = persistResult.hasPersistError;
    const whereTo = getFirstAvailableOfCollectionLoop(loopName, factGraph, newId, flow);

    if (!whereTo) throw new Error(`Cannot find a next path`);
    if (!hasPersistError) {
      navigate(whereTo.routable.fullRoute(whereTo.collectionId, { reviewMode: false }));
    }
  }, [collectionItems, factGraph, concretePath, saveAndPersist, loopName, flow, navigate]);

  const collectionDone = donePath
    ? factGraph.get(Path.concretePath(donePath, null)).complete && factGraph.get(Path.concretePath(donePath, null)).get
    : false;

  const collectionCompletion = collectionItems.map((itemId) => {
    return loop?.collectionItemCompletedCondition
      ? new Condition(loop.collectionItemCompletedCondition).evaluate(factGraph, itemId)
      : true;
  });
  // go through collectionItems to see if any items incomplete
  const hasIncompleteCollectionItem = collectionCompletion.some((isComplete) => !isComplete);
  const hasSeenReviewScreen = factGraph.get(Path.concretePath(`/hasSeenReviewScreen`, null));
  const shouldSeeHubCompletionButtons = loop?.shouldSeeHubCompletionBtnsPath
    ? factGraph.get(Path.concretePath(loop.shouldSeeHubCompletionBtnsPath, null)).get
    : true;
  const reviewUnlocked = hasSeenReviewScreen.complete ? hasSeenReviewScreen.get : false;

  return (
    <div className='screen__page-actions'>
      {isReturnEditable && (
        <Button type='button' outline onClick={create}>
          <Icon.Add size={3} className='usa-button__icon-left' aria-hidden='true' />
          {t(`fields.${path}.controls.add`)}
        </Button>
      )}
      {shouldSeeHubCompletionButtons ? (
        <div className='screen__actions'>
          {loopName && !collectionDone && isReturnEditable ? (
            // I don't have or am done adding button, seen on first edit
            <Button type='button' onClick={complete}>
              {t(`fields.${path}.controls.complete`, { count: collectionItems.length })}
              <Icon.NavigateNext size={3} className='usa-button__icon-right' aria-hidden='true' />
            </Button>
          ) : (
            <>
              {/* On review button, go up one level (unless knocked) */}
              <ButtonsUp loop={loop} navigate={navigate} knockoutRoute={getIsKnockedOut() && knockoutRoute} t={t} />
              {reviewUnlocked && !hasIncompleteCollectionItem && (
                <Button
                  type='button'
                  outline
                  className='margin-top-0'
                  onClick={() => goToKnockoutOrNavigate(`/flow/complete/review/review`)}
                >
                  {t(`button.reviewAndConfirm`)}
                  <Icon.NavigateFarNext size={3} className='usa-button__icon-right' aria-hidden='true' />
                </Button>
              )}
            </>
          )}
        </div>
      ) : (
        <></>
      )}
    </div>
  );
};
