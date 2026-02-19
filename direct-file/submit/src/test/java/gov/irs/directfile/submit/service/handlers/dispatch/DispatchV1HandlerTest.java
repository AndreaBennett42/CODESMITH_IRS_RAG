package gov.irs.directfile.submit.service.handlers.dispatch;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.models.Dispatch;
import gov.irs.directfile.models.message.MessageHeaderAttribute;
import gov.irs.directfile.models.message.QueueMessageHeaders;
import gov.irs.directfile.models.message.dispatch.DispatchMessageVersion;
import gov.irs.directfile.models.message.dispatch.VersionedDispatchMessage;
import gov.irs.directfile.models.message.dispatch.payload.AbstractDispatchPayload;
import gov.irs.directfile.models.message.dispatch.payload.DispatchPayloadV1;
import gov.irs.directfile.submit.domain.UserSubmission;
import gov.irs.directfile.submit.service.UserSubmissionBatchAssembler;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class DispatchV1HandlerTest {
    @Mock
    private UserSubmissionBatchAssembler batchAssembler;

    private DispatchV1Handler handler;
    private VersionedDispatchMessage<AbstractDispatchPayload> queueMessage;
    private Dispatch testDispatch;

    @BeforeEach
    public void setup() {
        handler = new DispatchV1Handler(batchAssembler);

        testDispatch = Dispatch.testObjectFactory();
        AbstractDispatchPayload payload = new DispatchPayloadV1(testDispatch);
        queueMessage = new VersionedDispatchMessage<>(
                payload,
                new QueueMessageHeaders()
                        .addHeader(MessageHeaderAttribute.VERSION, DispatchMessageVersion.V1.getVersion()));
    }

    @Test
    public void handleDispatchMessage_success() throws Exception {
        handler.handleDispatchMessage(queueMessage);

        UserSubmission userSubmission = UserSubmission.fromDispatch(testDispatch);
        verify(batchAssembler, times(1)).addSubmission(eq(userSubmission));
    }
}
