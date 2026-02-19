package gov.irs.directfile.api.taxreturn.submissions;

import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.*;
import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.pdfBackfill.PDFBackfillToS3Handler;
import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.publishsc.PublishSubmissionConfirmationsEventHandler;
import gov.irs.directfile.models.message.S3NotificationEvent;

@Service
@Slf4j
public class S3NotificationEventRouter {
    private static final Map<S3NotificationEvent, S3NotificationEventHandler> handlers = new HashMap<>();

    public S3NotificationEventRouter(
            TechnicalErrorResolvedHandler technicalErrorResolvedHandler,
            ReminderEmailHandler reminderEmailHandler,
            PDFBackfillToS3Handler pdfBackfillToS3Handler,
            PublishDispatchMessageEventHandler publishDispatchMessageEventHandler,
            PublishSubmissionConfirmationsEventHandler publishSubmissionConfirmationsEventHandler) {
        handlers.put(S3NotificationEvent.TECHNICAL_ERROR_RESOLVED, technicalErrorResolvedHandler);
        handlers.put(S3NotificationEvent.REMINDER_EMAIL, reminderEmailHandler);
        handlers.put(S3NotificationEvent.BACKFILL_HISTORICAL_PDFS, pdfBackfillToS3Handler);
        handlers.put(S3NotificationEvent.PUBLISH_DISPATCH_QUEUE_MESSAGES, publishDispatchMessageEventHandler);
        handlers.put(S3NotificationEvent.PUBLISH_SUBMISSION_CONFIRMATIONS, publishSubmissionConfirmationsEventHandler);
    }

    public void routeMessage(JsonNode message) {
        S3NotificationEvent key = S3NotificationEvent.getEnum(message.get("key").asText());
        JsonNode payload = message.get("payload");
        S3NotificationEventHandler handler = handlers.get(key);
        handler.handleNotificationEvent(payload);
    }
}
