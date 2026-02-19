package gov.irs.directfile.status.services;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.PublisherException;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.status.StatusChangeMessageVersion;
import gov.irs.directfile.models.message.status.VersionedStatusChangeMessage;
import gov.irs.directfile.models.message.status.payload.AbstractStatusChangePayload;
import gov.irs.directfile.models.message.status.payload.StatusChangePayloadV1;

@Service
@Slf4j
public class StatusChangeMessageService {
    private final List<StatusChangePublisher> publishers;
    private final ObjectMapper mapper;

    public StatusChangeMessageService(List<StatusChangePublisher> publishers, ObjectMapper mapper) {
        this.publishers = publishers;
        this.mapper = mapper;
    }

    public void publishStatusChangePayloadV1(Map<String, List<String>> statusSubmissionIdMap) {
        AbstractStatusChangePayload payload = new StatusChangePayloadV1(statusSubmissionIdMap);
        publishStatusChangePayload(payload, StatusChangeMessageVersion.V1);
    }

    private void publishStatusChangePayload(AbstractStatusChangePayload payload, StatusChangeMessageVersion version) {
        if (publishers == null || publishers.isEmpty()) {
            return;
        }

        VersionedStatusChangeMessage<AbstractStatusChangePayload> message = new VersionedStatusChangeMessage<>(
                payload, new QueueMessageHeaders().addHeader(MessageHeaderAttribute.VERSION, version.getVersion()));

        String jsonString;
        try {
            jsonString = mapper.writeValueAsString(message);
        } catch (JsonProcessingException e) {
            String errorMessage = "Exception calling writeValueAsString";
            log.error(errorMessage, e);
            throw new PublisherException(errorMessage, e);
        }

        List<String> errors = new ArrayList<>();
        for (StatusChangePublisher publisher : publishers) {
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
