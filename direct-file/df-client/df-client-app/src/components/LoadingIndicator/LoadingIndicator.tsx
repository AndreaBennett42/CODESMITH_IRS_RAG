/**
 * This component is based on the Loading Indicator from
 * the Department of Veterans Affairs' Design System
 * https://github.com/department-of-veterans-affairs/component-library/
 */
import { useEffect, useState } from 'react';
import Translation from '../Translation/index.js';
import styles from './LoadingIndicator.module.scss';
import classNames from 'classnames';

type LoadingIndicatorProps = {
  i18nKey?: string;
  delayMS?: number;
  inline?: boolean;
};

const ONE_SECOND = 1000;

const LoadingIndicator = ({
  i18nKey = `loadingIndicator.loading`,
  delayMS = ONE_SECOND,
  inline,
}: LoadingIndicatorProps) => {
  const [isLiveRegionReady, setIsLiveRegionReady] = useState(false);

  const containerClasses = classNames(`${styles.loadingIndicatorContainer}`, {
    [`${styles.loadingIndicatorContainerInline}`]: inline,
  });

  const spinnerClasses = classNames(`${styles.loadingIndicator}`, {
    [`${styles.loadingIndicatorInline} usa-button__icon-left`]: inline,
  });

  useEffect(() => {
    // show after a delay, to help prevent flickering
    const timeout = setTimeout(() => {
      /* There are two things going on here:
       *
       * 1. important! screen readers won't announce the inner content
       * unless the live region is rendered before the inner content
       * so we `useEffect` with an empty array to set setIsLiveRegionReady
       * after the initial render completes
       *
       * 2. showing the loading indicator instantly can cause annoying
       * flickering and make page load _feel_ slower, so we `setTimeout`
       * and show the indicator after a delay
       */
      setIsLiveRegionReady(true);
    }, delayMS);

    return () => clearTimeout(timeout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div role='status' aria-atomic='true' aria-live='polite' className={containerClasses}>
      {isLiveRegionReady && (
        <>
          <div className={spinnerClasses} aria-hidden='true' />
          <div className={inline ? `usa-sr-only` : undefined}>
            <Translation i18nKey={i18nKey} collectionId={null} />
          </div>
        </>
      )}
    </div>
  );
};

export default LoadingIndicator;
