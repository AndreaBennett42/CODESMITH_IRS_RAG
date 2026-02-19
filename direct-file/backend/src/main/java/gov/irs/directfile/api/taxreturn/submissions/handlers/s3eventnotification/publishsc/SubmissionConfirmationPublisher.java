package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.publishsc;

import gov.irs.directfile.models.message.Publisher;

public interface SubmissionConfirmationPublisher extends Publisher {
    // Just a marker interface to allow Spring to inject all the publishers for this message type
}
