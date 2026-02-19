package gov.irs.directfile.models.email;

import lombok.Getter;

@Getter
public enum HtmlTemplate {
    ACCEPTED,
    REJECTED,
    SUBMITTED,
    PRE_SUBMISSION_ERROR,
    POST_SUBMISSION_ERROR,
    ERROR_RESOLVED,
    REMINDER_SUBMIT,
    REMINDER_RESUBMIT,
    REMINDER_STATE,
    NON_COMPLETION_SURVEY;
}
