package gov.irs.directfile.models.message;

import lombok.Getter;

import gov.irs.directfile.models.message.exception.UnsupportedVersionException;

@Getter
@SuppressWarnings("PMD.UnnecessaryFullyQualifiedName")
public enum S3NotificationEvent {
    TECHNICAL_ERROR_RESOLVED("technical_error_resolved"),
    REMINDER_EMAIL("reminder_email"),
    BACKFILL_HISTORICAL_PDFS("backfill_pdfs"),
    PUBLISH_DISPATCH_QUEUE_MESSAGES("publish_dispatch_queue_messages"),
    RETRANSMIT_FOR_ANALYTICS("retransmit_for_analytics"),
    PUBLISH_SUBMISSION_CONFIRMATIONS("publish_submission_confirmations");

    private final String eventType;

    S3NotificationEvent(String eventType) {
        this.eventType = eventType;
    }

    public static S3NotificationEvent getEnum(String eventType) {
        if (TECHNICAL_ERROR_RESOLVED.getEventType().equals(eventType)) {
            return S3NotificationEvent.TECHNICAL_ERROR_RESOLVED;
        } else if (REMINDER_EMAIL.getEventType().equals(eventType)) {
            return S3NotificationEvent.REMINDER_EMAIL;
        } else if (BACKFILL_HISTORICAL_PDFS.getEventType().equals(eventType)) {
            return S3NotificationEvent.BACKFILL_HISTORICAL_PDFS;
        } else if (PUBLISH_DISPATCH_QUEUE_MESSAGES.getEventType().equals(eventType)) {
            return S3NotificationEvent.PUBLISH_DISPATCH_QUEUE_MESSAGES;
        } else if (RETRANSMIT_FOR_ANALYTICS.getEventType().equals(eventType)) {
            return S3NotificationEvent.RETRANSMIT_FOR_ANALYTICS;
        } else if (PUBLISH_SUBMISSION_CONFIRMATIONS.getEventType().equals(eventType)) {
            return S3NotificationEvent.PUBLISH_SUBMISSION_CONFIRMATIONS;
        } else {
            throw new UnsupportedVersionException("No enum found for eventType: " + eventType);
        }
    }
}
