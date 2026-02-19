package gov.irs.directfile.models.message.event;

import com.fasterxml.jackson.annotation.JsonValue;
import lombok.Getter;

/**
 * The SubmissionEventType enum represents the states a TaxReturnSubmission takes as it moves through our backend systems.
 * It is an internal only representation of our system at various points of time, and is not exposed to end users.
 * - A Processing event type indicates that a return is submitted but we haven't confirmed submission to MeF
 * - A Submitted event type indicates that MeF has confirmed submission of the return
 * - A Accepted event type indicates that MeF has accepted the return
 * - A Rejected event type indicates that MeF has rejected the return
 * - A Failed event type indicates that an error occurred while processing the submission such that it never reached
 * the state it was supposed to.
 */
@Getter
@SuppressWarnings("PMD.UnnecessaryFullyQualifiedName")
public enum SubmissionEventTypeEnum {
    PROCESSING("Processing"),
    SUBMITTED("Submitted"),
    ACCEPTED("Accepted"),
    REJECTED("Rejected"),
    FAILED("Failed"),
    FAILURE_RESOLVED("FailureResolved"),
    PRE_SUBMISSION_ERROR("PreSubmissionError"),
    POST_SUBMISSION_ERROR("PostSubmissionError"),
    ERROR_RESOLVED("ErrorResolved"),
    REMINDER_SUBMIT("ReminderSubmit"),
    REMINDER_RESUBMIT("ReminderResubmit"),
    REMINDER_STATE_TAX("ReminderStateTax");

    @JsonValue
    final String eventType;

    SubmissionEventTypeEnum(String eventType) {
        this.eventType = eventType;
    }

    public static SubmissionEventTypeEnum getEnum(String eventType) {
        for (SubmissionEventTypeEnum submissionEventTypeEnum : SubmissionEventTypeEnum.values()) {
            if (submissionEventTypeEnum.eventType.equalsIgnoreCase(eventType)) return submissionEventTypeEnum;
        }
        throw new IllegalArgumentException("No enum found for submission event type: " + eventType);
    }

    @Override
    public String toString() {
        return eventType;
    }
}
