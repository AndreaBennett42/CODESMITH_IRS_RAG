package gov.irs.directfile.api.taxreturn.submissions.handlers.status;

import java.util.ArrayList;
import java.util.List;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.taxreturn.submissions.ConfirmationService;
import gov.irs.directfile.api.taxreturn.submissions.SubmissionStatusesMessage;
import gov.irs.directfile.models.email.HtmlTemplate;
import gov.irs.directfile.models.message.status.VersionedStatusChangeMessage;
import gov.irs.directfile.models.message.status.payload.AbstractStatusChangePayload;
import gov.irs.directfile.models.message.status.payload.StatusChangePayloadV1;

@Service
@Slf4j
@AllArgsConstructor
public class StatusChangeV1Handler implements StatusChangeHandler {
    private final ConfirmationService confirmationService;

    @Override
    public void handleStatusChangeMessage(VersionedStatusChangeMessage<AbstractStatusChangePayload> message) {
        StatusChangePayloadV1 payload = (StatusChangePayloadV1) message.getPayload();
        List<SubmissionStatusesMessage> submissionStatuses = convertMessageToSubmissionStatuses(payload);

        confirmationService.handleStatusChangeEvents(submissionStatuses);
    }

    protected List<SubmissionStatusesMessage> convertMessageToSubmissionStatuses(StatusChangePayloadV1 payload) {
        List<SubmissionStatusesMessage> submissionStatuses = new ArrayList<>();

        for (var entry : payload.getStatusSubmissionIdMap().entrySet()) {
            submissionStatuses.add(new SubmissionStatusesMessage(
                    HtmlTemplate.valueOf(entry.getKey().toUpperCase()), entry.getValue()));
        }

        return submissionStatuses;
    }
}
