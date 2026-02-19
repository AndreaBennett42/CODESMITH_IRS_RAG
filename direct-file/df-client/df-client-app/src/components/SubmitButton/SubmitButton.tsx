import { Button } from '@trussworks/react-uswds';
import { FC, useContext, useEffect, useCallback, useState, SyntheticEvent } from 'react';
import { NetworkConnectionContext } from '../../context/networkConnectionContext.js';
import Translation from '../Translation/index.js';
import { ScreenButtonProps } from '../../types/core.js';
import { checkAllFormControlValidity } from '../../misc/constraintsApi.js';
import { scrollToScreenHeader, scrollToTop } from '../../misc/misc.js';
import usePrefersReducedMotion from '../../hooks/usePrefersReducedMotion.js';
import classnames from 'classnames';
import { getTaxReturnById } from '../../utils/taxReturnUtils.js';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';
import LoadingIndicator from '../LoadingIndicator/LoadingIndicator.js';
import { useSubmit } from '../../hooks/useSubmit.js';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import { CompleteFactGraphResult } from '@irs/js-factgraph-scala/src/typings/FactGraph.js';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { Path } from '../../flow/Path.js';

const SubmitButton: FC<ScreenButtonProps> = ({
  gotoNextScreen,
  collectionId,
  setShowFeedback,
  focusOnErrorOrSummary,
  i18nKey = `button.submit`,
}) => {
  const { online } = useContext(NetworkConnectionContext);
  const { taxReturns, fetchTaxReturns, currentTaxReturnId } = useContext(TaxReturnsContext);
  const { isFetching: isFetchingSubmissionStatus, fetchSubmissionStatus } = useContext(SubmissionStatusContext);
  const prefersReducedMotion = usePrefersReducedMotion();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isDisabled, setIsDisabled] = useState(false);
  const currentTaxReturn = getTaxReturnById(taxReturns, currentTaxReturnId);
  const [hasLatestReturns, setHasLatestReturns] = useState(false);
  const { factGraph } = useFactGraph();
  const signAndSubmit = useSubmit();

  const isResubmittingFactResult = factGraph.get(Path.concretePath(`/isResubmitting`, null));
  const isResubmitting =
    isResubmittingFactResult.complete && (isResubmittingFactResult as CompleteFactGraphResult<boolean>).get;

  // TODO - this kind of code is a bug factory. We need to have a real network syncronization layer
  // instead of a mess of useEffect code.
  useEffect(() => {
    if (!hasLatestReturns) {
      fetchTaxReturns();
      setHasLatestReturns(true);
    }
  }, [fetchTaxReturns, hasLatestReturns, setHasLatestReturns]);

  useEffect(() => {
    function confirmClose(e: Event) {
      e.preventDefault();
    }
    isSubmitting && window.addEventListener(`beforeunload`, confirmClose);
    return () => {
      isSubmitting && window.removeEventListener(`beforeunload`, confirmClose);
    };
  }, [isSubmitting]);

  const buttonClasses = classnames({
    'usa-button--disabled': isSubmitting || isDisabled || !currentTaxReturn?.isEditable,
  });

  const handleClick = useCallback(
    async (e: SyntheticEvent) => {
      e.preventDefault();
      // Immediately prevent additional calls if a submission attempt is underway
      if (isSubmitting) {
        return;
      }

      setIsSubmitting(true);

      const allFieldsValid = checkAllFormControlValidity(`the_form`);
      if (!allFieldsValid) {
        setIsSubmitting(false);
        setShowFeedback(true);
        focusOnErrorOrSummary();
        return;
      }

      if (!online) {
        setIsSubmitting(false);
        scrollToTop(prefersReducedMotion);
        return;
      }

      setShowFeedback(false);
      const { hasSubmitError, isRetryDisabled } = await signAndSubmit();

      const refreshStatus = () => {
        if (currentTaxReturnId && !isFetchingSubmissionStatus) {
          fetchSubmissionStatus(currentTaxReturnId);
        }
      };

      setIsSubmitting(false);
      setIsDisabled(isRetryDisabled);
      if (hasSubmitError) {
        scrollToScreenHeader();
      } else {
        if (isResubmitting) {
          refreshStatus();
        }
        gotoNextScreen();
      }

      // After submitting, successful or not, refresh tax returns so that client and backend are in sync
      fetchTaxReturns();
    },
    [
      isSubmitting,
      online,
      prefersReducedMotion,
      currentTaxReturnId,
      isResubmitting,
      isFetchingSubmissionStatus,
      fetchSubmissionStatus,
      fetchTaxReturns,
      focusOnErrorOrSummary,
      gotoNextScreen,
      setShowFeedback,
      signAndSubmit,
    ]
  );

  return (
    <div className='screen__actions'>
      <Button className={buttonClasses} type='submit' onClick={handleClick}>
        {isSubmitting ? (
          <>
            <LoadingIndicator inline delayMS={0} />
            <Translation i18nKey={i18nKey} collectionId={collectionId} />
          </>
        ) : (
          <Translation i18nKey={i18nKey} collectionId={collectionId} />
        )}
      </Button>
    </div>
  );
};

export default SubmitButton;
