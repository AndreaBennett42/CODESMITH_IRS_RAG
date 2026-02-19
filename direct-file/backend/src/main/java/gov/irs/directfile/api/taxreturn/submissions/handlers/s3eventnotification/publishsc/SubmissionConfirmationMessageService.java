package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.publishsc;

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.PublisherException;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.confirmation.SubmissionConfirmationMessageVersion;
import gov.irs.directfile.models.message.confirmation.VersionedSubmissionConfirmationMessage;
import gov.irs.directfile.models.message.confirmation.payload.AbstractSubmissionConfirmationPayload;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;

@Service
@Slf4j
public class SubmissionConfirmationMessageService {
    private final List<SubmissionConfirmationPublisher> publishers;
    private final ObjectMapper mapper;

    public SubmissionConfirmationMessageService(List<SubmissionConfirmationPublisher> publishers, ObjectMapper mapper) {
        this.publishers = publishers;
        this.mapper = mapper;
    }

    public void publishSubmissionConfirmationPayloadV2(List<SubmissionConfirmationPayloadV2Entry> entries) {
        AbstractSubmissionConfirmationPayload payload = new SubmissionConfirmationPayloadV2(entries);
        publishSubmissionConfirmationPayload(payload, SubmissionConfirmationMessageVersion.V2);
    }

    private void publishSubmissionConfirmationPayload(
            AbstractSubmissionConfirmationPayload payload, SubmissionConfirmationMessageVersion version) {
        if (publishers == null || publishers.isEmpty()) {
            return;
        }

        VersionedSubmissionConfirmationMessage<AbstractSubmissionConfirmationPayload> message =
                new VersionedSubmissionConfirmationMessage<>(
                        payload,
                        new QueueMessageHeaders().addHeader(MessageHeaderAttribute.VERSION, version.getVersion()));

        String jsonString;
        try {
            jsonString = mapper.writeValueAsString(message);
        } catch (JsonProcessingException e) {
            String errorMessage = "Exception calling writeValueAsString";
            log.error(errorMessage, e);
            throw new PublisherException(errorMessage, e);
        }

        List<String> errors = new ArrayList<>();
        for (SubmissionConfirmationPublisher publisher : publishers) {
            try {
                publisher.publish(jsonString);
            } catch (Exception e) {
                log.error("Exception calling publish", e);
                errors.add(e.getMessage() + " (" + publisher.getClass().getSimpleName() + ")");
            }
        }

        if (!errors.isEmpty()) {
            String errorMessage = StringUtils.join(errors, "; ");
            log.error(errorMessage);
            throw new PublisherException(errorMessage);
        }
    }
}
