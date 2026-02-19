import { MouseEventHandler, SyntheticEvent, useCallback, useState } from 'react';
import { Button, Icon } from '@trussworks/react-uswds';
import { Condition } from '../../flow/Condition.js';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import Translation from '../Translation/index.js';
import { useNavigate, useParams } from 'react-router-dom';
import { Path } from '../../flow/Path.js';
import { SaveAndOrContinueAndSetFactButtonProps } from '../../types/core.js';
import { useChecklistState } from '../../flow/useChecklistState.js';
import DFAlert from '../Alert/DFAlert.js';
import { useSaveAndPersistIfPossible } from '../../hooks/useSaveAndPersistIfPossible.js';
type TheActualButtonProps = {
  disabledIfPressedClass: string | undefined;
  handleClick: MouseEventHandler<HTMLButtonElement>;
  i18nKey: SaveAndOrContinueAndSetFactButtonProps[`i18nKey`];
  collectionId: SaveAndOrContinueAndSetFactButtonProps[`collectionId`];
};

/**
 * We do not want to call useChecklistState for every Screen using this button when only those that might block on error
 * Since  you cannot conditionally call hooks, but you can conditionally render components, this "component" is just a
 * wrapper for the button, which also calls useChecklistState if it is being rendered to determine if it should not
 * render the button
 */
const MaybeTheActualButtonOrBlockingAlert = (buttonProps: TheActualButtonProps) => {
  const params = useParams();
  const currentUrl = `/flow/${params[`*`]}`;
  const subcategoryRoute = currentUrl.substring(0, currentUrl.lastIndexOf(`/`));
  const checklistState = useChecklistState();
  const subcategories = checklistState.flatMap((c) => c.subcategories);
  const indexOfCurrentSubcategory = subcategories.findIndex((sc) => sc.subcategoryRoute === subcategoryRoute);
  const subcategoriesBeforeCurrentSubcategory = subcategories.slice(0, indexOfCurrentSubcategory);
  const hasAtLeastOneErrorOnAnySubCategory = subcategoriesBeforeCurrentSubcategory.some(
    (sc) => [...sc.mefAlertConfigs.errors, ...sc.alertConfigs.errors].length > 0
  );
  const hasAtLeastOneIncompleteAnySubCategory = subcategoriesBeforeCurrentSubcategory.some(
    (sc) => !sc.isComplete || sc.hasIncompletedCollectionItem
  );
  return hasAtLeastOneErrorOnAnySubCategory || hasAtLeastOneIncompleteAnySubCategory ? (
    <DFAlert
      type='error'
      i18nKey='/info/complete/review/review-and-confirm/info-alert'
      headingLevel='h2'
      collectionId={null}
    />
  ) : (
    <TheActualButton {...buttonProps} />
  );
};

const TheActualButton = ({ disabledIfPressedClass, handleClick, i18nKey, collectionId }: TheActualButtonProps) => {
  return (
    <Button className={disabledIfPressedClass} type='submit' onClick={handleClick}>
      <Translation i18nKey={i18nKey ? i18nKey : ``} collectionId={collectionId} />
      <Icon.NavigateNext size={3} className='usa-button__icon-right' aria-hidden='true' />
    </Button>
  );
};

const SaveAndOrContinueAndSetFactButton: React.FC<SaveAndOrContinueAndSetFactButtonProps> = ({
  gotoNextScreen,
  enabled,
  i18nKey,
  collectionId,
  nextRouteOverride,
  sourcePath,
  destinationPath,
  shouldBlockOnErrors = false,
}) => {
  const { factGraph } = useFactGraph();
  const navigate = useNavigate();
  const isDisabled = enabled && !new Condition(enabled).evaluate(factGraph, collectionId);
  const saveAndPersistIfPossible = useSaveAndPersistIfPossible();
  const [isSaving, setIsSaving] = useState(false);
  const disabledIfPressedClass = isDisabled || isSaving ? `usa-button--disabled` : undefined;
  const handleClick = useCallback(
    async (event: SyntheticEvent) => {
      event.preventDefault();

      // Prevent additional calls if Enter key is pressed
      if (isSaving) {
        return;
      }

      // First save any value that might have been manually entered.
      // eslint-disable-next-line df-rules/no-factgraph-save
      factGraph.save();

      if (sourcePath && destinationPath) {
        const concreteSourcePath = Path.concretePath(sourcePath, collectionId);
        const concreteDestinationPath = Path.concretePath(destinationPath, collectionId);
        const maybeValue = factGraph.get(concreteSourcePath);
        if (maybeValue.hasValue) {
          factGraph.set(concreteDestinationPath, maybeValue.get);
        }
      }
      setIsSaving(true);
      const saveResult = await saveAndPersistIfPossible();
      if (!saveResult.hasPersistError) nextRouteOverride ? navigate(nextRouteOverride) : gotoNextScreen();

      setIsSaving(false);
    },
    [
      isSaving,
      sourcePath,
      destinationPath,
      saveAndPersistIfPossible,
      nextRouteOverride,
      navigate,
      gotoNextScreen,
      collectionId,
      factGraph,
    ]
  );

  const buttonProps: TheActualButtonProps = {
    disabledIfPressedClass,
    handleClick,
    i18nKey,
    collectionId,
  };

  return (
    <div className='screen__actions screen__actions--no-border'>
      {shouldBlockOnErrors ? (
        <MaybeTheActualButtonOrBlockingAlert {...buttonProps} />
      ) : (
        <TheActualButton {...buttonProps} />
      )}
    </div>
  );
};

export default SaveAndOrContinueAndSetFactButton;
