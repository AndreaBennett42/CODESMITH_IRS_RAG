import useFact from '../../hooks/useFact.js';
import AggregateSummaryAlert, { AggregateSummaryAlertProps } from '../SummaryAlert/AggregateSummaryAlert.js';

import FederalReturnStatusAlert from '../FederalReturnStatusAlert/FederalReturnStatusAlert.js';
import { useContext } from 'react';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import { getTaxReturnById, hasBeenSubmitted } from '../../utils/taxReturnUtils.js';
import PaperPathStatusAlert from '../PaperPathStatusAlert/PaperPathStatusAlert.js';
import { Path } from '../../flow/Path.js';
import SummaryAlert from '../SummaryAlert/SummaryAlert.js';
import { useSystemAlertContext } from '../../context/SystemAlertContext/SystemAlertContext.js';
import SystemAlertAggregator from '../SystemAlertAggregator/SystemAlertAggregator.js';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';
import { useKnockoutCheck } from '../../hooks/useKnockoutCheck.js';

export type SectionsAlertAggregatorProps = {
  showStatusAlert?: boolean;
  summaryErrorSections: AggregateSummaryAlertProps[`summaryErrorSections`];
  summaryWarningSections: AggregateSummaryAlertProps[`summaryWarningSections`];
  refs: AggregateSummaryAlertProps[`refs`];
  separateAlertSummariesByType?: boolean;
  collectionName?: string;
  collectionId?: string | null;
};

/* eslint-disable max-len */
/**
 * A component that deals with presentation of alerts following the DF Alert Aggregation System:
 * - Wiki: https://git.irslabs.org/irslabs-prototypes/direct-file/-/wikis/Errors,-Warnings,-and-Status-Messages
 */
/* eslint-enable max-len */
const SectionsAlertAggregator = ({
  showStatusAlert = false,
  summaryErrorSections,
  summaryWarningSections,
  refs,
  separateAlertSummariesByType = false,
  collectionName = ``,
  collectionId = null,
}: SectionsAlertAggregatorProps) => {
  const { submissionStatus } = useContext(SubmissionStatusContext);
  const { taxReturns, currentTaxReturnId } = useContext(TaxReturnsContext);
  const { systemAlerts } = useSystemAlertContext();

  const currentTaxReturn = getTaxReturnById(taxReturns, currentTaxReturnId);
  const [isPaperPath] = useFact<boolean>(Path.concretePath(`/isPaperPath`, null));

  const hasSystemAlerts = Object.keys(systemAlerts).length > 0;
  const hasErrors = summaryErrorSections.length > 0;
  const hasWarnings = summaryWarningSections.length > 0;
  const shouldShowFederalReturnStatusAlert =
    showStatusAlert && currentTaxReturn && hasBeenSubmitted(currentTaxReturn) && submissionStatus;
  const shouldShowPaperPathStatusAlert = showStatusAlert && isPaperPath;
  const shouldShowSummaryAlerts = hasErrors || hasWarnings;
  const { getIsKnockedOut } = useKnockoutCheck();
  const isKnockedOut = getIsKnockedOut();

  const shouldRender =
    hasSystemAlerts || shouldShowFederalReturnStatusAlert || shouldShowPaperPathStatusAlert || shouldShowSummaryAlerts;

  return (
    shouldRender && (
      <div className='margin-top-3' data-testid='sections-alert-aggregator'>
        {
          // System alerts
          <SystemAlertAggregator />
        }
        {
          // Return status banner
          shouldShowFederalReturnStatusAlert && (
            <FederalReturnStatusAlert
              taxReturn={currentTaxReturn}
              submissionStatus={submissionStatus}
              data-testid='federal-return-status-alert'
            />
          )
        }
        {shouldShowPaperPathStatusAlert && <PaperPathStatusAlert />}
        {
          // Alert summary/summaries
          shouldShowSummaryAlerts &&
            (separateAlertSummariesByType ? (
              <>
                {
                  // Error summary
                  hasErrors && (
                    <SummaryAlert type='error' sections={summaryErrorSections} refs={refs} headingLevel='h3' />
                  )
                }
                {
                  // Warning summary
                  hasWarnings && (
                    <SummaryAlert type='warning' sections={summaryWarningSections} refs={refs} headingLevel='h3' />
                  )
                }
              </>
            ) : (
              !isKnockedOut && (
                <AggregateSummaryAlert
                  collectionName={collectionName}
                  collectionId={collectionId}
                  summaryErrorSections={summaryErrorSections}
                  summaryWarningSections={summaryWarningSections}
                  refs={refs}
                  headingLevel='h3'
                  data-testid='aggregate-summary-alert'
                />
              )
            ))
        }
      </div>
    )
  );
};

export default SectionsAlertAggregator;
