package gov.irs.directfile.api.taxreturn.models;

import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

/**
 * The SubmissionEventType enum represents the states a TaxReturnSubmission takes as it moves through our backend systems.
 * It is an internal only representation of our system at various points of time, and is not exposed to end users.
 * - A Processing event type indicates that a return is submitted but we haven't confirmed submission to MeF
 * - A Submitted event type indicates that MeF has confirmed submission of the return
 * - A Accepted event type indicates that MeF has accepted the return
 * - A Rejected event type indicates that MeF has rejected the return
 * - A Failed event type indicates that an error occurred while processing the submission such that it never reached
 * the state it was supposed to.
 *
 * @deprecated
 * <p>Use {@link SubmissionEventTypeEnum} instead.
 */
@Deprecated
public enum SubmissionEventType {
    Processing,
    Submitted,
    Accepted,
    Rejected,
    Failed,
    FailureResolved,
}
