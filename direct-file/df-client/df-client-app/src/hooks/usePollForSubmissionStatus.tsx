import { useContext, useEffect, useRef, useState } from 'react';
import { SubmissionStatusContext } from '../context/SubmissionStatusContext/SubmissionStatusContext.js';
import { TaxReturn } from '../types/core.js';
import { TaxReturnsContext } from '../context/TaxReturnsContext.js';
import { FEDERAL_RETURN_STATUS } from '../constants/taxConstants.js';
import { assertNever } from 'assert-never';

// For local testing, altering these settings can be handy
const POLLING_INTERVAL_MILLISECONDS = 60 * 1000;
const MAXIMUM_POLLING_ATTEMPTS = 10;

export type SubmissionStatusPollResult = { hasFinishedPolling: boolean; numPollsAttempted: number };

/**
 * After submission the status on the backend is not immediately pending.
 * This utility polls tax returns and status endpoints until such time as the actual, up-to-date status is retrieved.
 * @param taxReturn the tax return for which to poll
 * @param pollingIntervalMs the time between attempts to fetch tax return and/or status data
 * @param maximumPollingAttempts the maximum number of requests to make before giving up on polling
 */
export const usePollForSubmissionStatus = (
  taxReturn: TaxReturn,
  pollingIntervalMs: number = POLLING_INTERVAL_MILLISECONDS,
  maximumPollingAttempts: number = MAXIMUM_POLLING_ATTEMPTS
): SubmissionStatusPollResult => {
  const pollingTimerRef = useRef<NodeJS.Timeout>();
  const { fetchTaxReturns } = useContext(TaxReturnsContext);
  const { submissionStatus: retrievedSubmissionStatus, fetchSubmissionStatus } = useContext(SubmissionStatusContext);
  const [hasFinishedPolling, setHasFinishedPolling] = useState(false);
  const [numPollsAttempted, setNumPollsAttempted] = useState(0);

  useEffect(() => {
    const refreshStatus = () => {
      fetchSubmissionStatus(taxReturn.id);
      setNumPollsAttempted((currentPollingAttempts) => currentPollingAttempts + 1);
    };

    const startPollingStatus = () => {
      clearInterval(pollingTimerRef.current);
      pollingTimerRef.current = setInterval(refreshStatus, pollingIntervalMs);
    };

    const stopPolling = () => {
      clearInterval(pollingTimerRef.current);
    };

    // If the backend gave us a status, assume is correct
    if (retrievedSubmissionStatus) {
      const { status } = retrievedSubmissionStatus;
      if (status === FEDERAL_RETURN_STATUS.PENDING) {
        // If pending, we need to poll
        if (numPollsAttempted < maximumPollingAttempts) {
          setHasFinishedPolling(false);
          startPollingStatus();
        } else {
          setHasFinishedPolling(true);
          stopPolling();
        }
      } else {
        if (
          status !== FEDERAL_RETURN_STATUS.ACCEPTED &&
          status !== FEDERAL_RETURN_STATUS.REJECTED &&
          status !== FEDERAL_RETURN_STATUS.ERROR
        ) {
          // Ensure that if a new non-final status were to be added (e.g. SUBMITTED),
          // that this logic would need to be updated simultaneously
          assertNever(status);
        }
        stopPolling();
        setHasFinishedPolling(true);
      }
    }

    return () => {
      stopPolling();
    };
  }, [
    pollingIntervalMs,
    fetchSubmissionStatus,
    fetchTaxReturns,
    retrievedSubmissionStatus,
    taxReturn,
    numPollsAttempted,
    maximumPollingAttempts,
  ]);

  return { hasFinishedPolling, numPollsAttempted };
};
