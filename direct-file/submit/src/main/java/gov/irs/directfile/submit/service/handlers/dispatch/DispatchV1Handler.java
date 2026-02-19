package gov.irs.directfile.submit.service.handlers.dispatch;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.Dispatch;
import gov.irs.directfile.models.message.dispatch.VersionedDispatchMessage;
import gov.irs.directfile.models.message.dispatch.payload.AbstractDispatchPayload;
import gov.irs.directfile.models.message.dispatch.payload.DispatchPayloadV1;
import gov.irs.directfile.submit.domain.UserSubmission;
import gov.irs.directfile.submit.service.UserSubmissionBatchAssembler;

@Service
@Slf4j
@AllArgsConstructor
public class DispatchV1Handler implements DispatchHandler {
    private final UserSubmissionBatchAssembler userSubmissionBatchAssembler;

    @Override
    public void handleDispatchMessage(VersionedDispatchMessage<AbstractDispatchPayload> message) throws Exception {
        DispatchPayloadV1 payload = (DispatchPayloadV1) message.getPayload();
        Dispatch dispatch = payload.getDispatch();

        UserSubmission userSubmission = UserSubmission.fromDispatch(dispatch);
        userSubmissionBatchAssembler.addSubmission(userSubmission);
    }
}
