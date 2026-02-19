package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification;

import com.fasterxml.jackson.databind.JsonNode;

public interface S3NotificationEventHandler {

    void handleNotificationEvent(JsonNode payload);
}
