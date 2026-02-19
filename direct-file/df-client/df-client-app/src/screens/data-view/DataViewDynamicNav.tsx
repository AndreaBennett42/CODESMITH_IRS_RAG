import { Alert } from '@trussworks/react-uswds';
import { Link, useSearchParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ScreenConfig } from '../../flow/ScreenConfig.js';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { Path } from '../../flow/Path.js';
import { DataViewButton } from './DataViewButton.js';
import { Condition } from '../../flow/Condition.js';
import { useFlow } from '../../flow/flowConfig.js';
import { conditionsPass } from '../../utils/condition.js';
import { useKnockoutCheck } from '../../hooks/useKnockoutCheck.js';
import styles from './SubSubCategory.module.scss';
import { findLast } from '../../utils/polyfills.js';

type DataViewDynamicNavProps = {
  route: string;
  showAlert: boolean | ScreenConfig | undefined;
  showNavButtons: boolean;
  sectionIsComplete: boolean;
  isSignAndSubmit: boolean;
};

export const DataViewDynamicNav = ({
  route,
  showAlert,
  showNavButtons,
  sectionIsComplete,
  isSignAndSubmit,
}: DataViewDynamicNavProps) => {
  const [searchParams] = useSearchParams();
  const { t } = useTranslation();
  const { factGraph } = useFactGraph();
  const flow = useFlow();
  const { getIsKnockedOut } = useKnockoutCheck();
  const isKnockedOut = getIsKnockedOut();
  const hasSeenReviewScreen = factGraph.get(Path.concretePath(`/hasSeenReviewScreen`, null));
  const isReviewAndConfirmUnlocked =
    hasSeenReviewScreen.complete && !getIsKnockedOut() ? hasSeenReviewScreen.get : false;
  const isReviewMode = searchParams.get(`reviewMode`);

  const categories = flow.categories;

  // We need to check the completeness of the return (excluding the submit section)
  // to make sure we haven't invalidated anything in the return before showing the button
  // shortcut to the last screen.
  const returnIsComplete = categories
    .flatMap((cat) =>
      cat.subcategories
        // Submit completion is always false, so filter it out.
        .filter(
          (sc) =>
            sc.route !== `/flow/complete/submit` &&
            (!sc.displayOnlyIf || Array.isArray(sc.displayOnlyIf)
              ? sc.displayOnlyIf?.some((condtion) => new Condition(condtion).evaluate(factGraph, null))
              : new Condition(sc.displayOnlyIf).evaluate(factGraph, null))
        )
        .every((subcat) => {
          const allCompletionConditions = Array.isArray(subcat.completionCondition)
            ? subcat.completionCondition
            : [subcat.completionCondition];
          return conditionsPass({ conditions: allCompletionConditions }, factGraph, null);
        })
    )
    .every((condition) => condition);

  // Get the last subcategory that's not hidden. This will be either "Print and Mail" or "Submit."
  const availableSubcategories = categories.find((cat) => cat.route === `/flow/complete`)?.subcategories;

  // I had to unwind this logic a little bit because of how typing interacts with the `?.` operator
  let lastAvailableSubcategory;
  if (availableSubcategories) {
    lastAvailableSubcategory = findLast(availableSubcategories, (subcat) => {
      if (!subcat.displayOnlyIf) {
        return false;
      } else if (Array.isArray(subcat.displayOnlyIf)) {
        return subcat.displayOnlyIf.some((condtion) => new Condition(condtion).evaluate(factGraph, null));
      } else {
        return new Condition(subcat.displayOnlyIf).evaluate(factGraph, null);
      }
    });
  }

  // Get the first available screen from the above.
  const firstScreenFromSubcategory = lastAvailableSubcategory?.screens.find((sc) => sc.isAvailable(factGraph, null));
  const buttonRoute = firstScreenFromSubcategory?.fullRoute(null) || `/checklist`;

  const showContinueButton = isSignAndSubmit && sectionIsComplete && returnIsComplete;
  const showReviewAndConfirmButton = isReviewAndConfirmUnlocked && showNavButtons && !isSignAndSubmit;
  const showScreenActions = showContinueButton || showNavButtons || showReviewAndConfirmButton;

  return (
    <>
      {showAlert && !isKnockedOut && (
        <div className={styles.dataViewAlert}>
          <Alert headingLevel='h2' type='error' slim>
            {t(`dataviews.incomplete`)}
            &nbsp;
            <Link to={route}>{t(`dataviews.resume`)}</Link>
          </Alert>
        </div>
      )}
      {showScreenActions && (
        <div className='screen__actions screen__actions--no-border'>
          {/* Continue button "shortcut" to the last screen of the return. */}
          {showContinueButton && <DataViewButton route={buttonRoute} isOutline={false} i18nKey={`button.continue`} />}
          {showNavButtons && (
            <DataViewButton
              route={`/checklist`}
              isOutline={!sectionIsComplete}
              i18nKey={
                isReviewMode || !sectionIsComplete || isSignAndSubmit ? `button.go-to-checklist` : `button.continue`
              }
            />
          )}
          {showReviewAndConfirmButton && (
            <DataViewButton
              route={`/flow/complete/review/review`}
              isOutline={true}
              i18nKey={`button.reviewAndConfirm`}
            />
          )}
        </div>
      )}
    </>
  );
};
