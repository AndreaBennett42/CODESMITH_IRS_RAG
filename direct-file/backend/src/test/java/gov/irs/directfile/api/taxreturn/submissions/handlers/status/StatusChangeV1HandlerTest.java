package gov.irs.directfile.api.taxreturn.submissions.handlers.status;

import java.util.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.api.taxreturn.submissions.ConfirmationService;
import gov.irs.directfile.api.taxreturn.submissions.SubmissionStatusesMessage;
import gov.irs.directfile.models.email.HtmlTemplate;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.status.StatusChangeMessageVersion;
import gov.irs.directfile.models.message.status.VersionedStatusChangeMessage;
import gov.irs.directfile.models.message.status.payload.AbstractStatusChangePayload;
import gov.irs.directfile.models.message.status.payload.StatusChangePayloadV1;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class StatusChangeV1HandlerTest {
    @Mock
    private ConfirmationService confirmationService;

    private StatusChangeV1Handler handler;
    private StatusChangePayloadV1 payloadV1;
    private VersionedStatusChangeMessage<AbstractStatusChangePayload> queueMessage;
    private List<SubmissionStatusesMessage> submissionStatuses;

    @BeforeEach
    public void setup() {
        handler = new StatusChangeV1Handler(confirmationService);

        // Use LinkedHashMap to preserve insertion order (so map can be easily tested for equality later)
        Map<String, List<String>> statusSubmissionIdMap = new LinkedHashMap<>();
        statusSubmissionIdMap.put("accepted", new ArrayList<>(List.of("123456789", "222222222")));
        statusSubmissionIdMap.put("rejected", new ArrayList<>(List.of("987654321")));
        payloadV1 = new StatusChangePayloadV1(statusSubmissionIdMap);

        submissionStatuses = new ArrayList<>();
        submissionStatuses.add(
                new SubmissionStatusesMessage(HtmlTemplate.valueOf("ACCEPTED"), List.of("123456789", "222222222")));
        submissionStatuses.add(new SubmissionStatusesMessage(HtmlTemplate.valueOf("REJECTED"), List.of("987654321")));

        queueMessage = new VersionedStatusChangeMessage<>(
                payloadV1,
                new QueueMessageHeaders()
                        .addHeader(MessageHeaderAttribute.VERSION, StatusChangeMessageVersion.V1.getVersion()));
    }

    @Test
    public void handleStatusChangeMessage_success() {
        handler.handleStatusChangeMessage(queueMessage);

        verify(confirmationService, times(1)).handleStatusChangeEvents(eq(submissionStatuses));
    }

    @Test
    public void convertMessageToSubmissionStatuses_success() {
        List<SubmissionStatusesMessage> result = handler.convertMessageToSubmissionStatuses(payloadV1);

        assertEquals(submissionStatuses, result);
    }
}
