package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification;

import java.util.*;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.transaction.Transactional;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.taxreturn.TaxReturnSubmissionRepository;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;
import gov.irs.directfile.api.taxreturn.submissions.ConfirmationService;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

@Service
@Slf4j
@Transactional
public class TechnicalErrorResolvedHandler implements S3NotificationEventHandler {

    private final ConfirmationService confirmationService;

    private final TaxReturnSubmissionRepository taxReturnSubmissionRepository;

    String PAYLOAD_KEY = "ids";

    public TechnicalErrorResolvedHandler(
            ConfirmationService confirmationService, TaxReturnSubmissionRepository taxReturnSubmissionRepository) {
        this.confirmationService = confirmationService;
        this.taxReturnSubmissionRepository = taxReturnSubmissionRepository;
    }

    @Override
    public void handleNotificationEvent(JsonNode payload) {
        sendTechnicalErrorResolvedEmails(payload);
    }

    protected void sendTechnicalErrorResolvedEmails(JsonNode payload) {
        Map<UUID, TaxReturnSubmission> taxReturnTaxReturnSubmissionMap = new HashMap<>();
        List<TaxReturnSubmission> taxReturnSubmissions = new ArrayList<>();
        Iterator<JsonNode> idsArr = payload.get(PAYLOAD_KEY).elements();

        idsArr.forEachRemaining(id -> {
            UUID taxReturnId = UUID.fromString(id.asText());
            Optional<TaxReturnSubmission> taxReturnSubmission =
                    taxReturnSubmissionRepository.findLatestTaxReturnSubmissionByTaxReturnId(taxReturnId);
            if (taxReturnSubmission.isPresent()) {
                TaxReturnSubmission trs = taxReturnSubmission.get();
                trs.addSubmissionEvent(SubmissionEventTypeEnum.ERROR_RESOLVED);
                taxReturnSubmissions.add(trs);
                taxReturnTaxReturnSubmissionMap.put(taxReturnId, trs);
            }
        });
        log.info(
                "Saving {} tax return submissions with a error resolved submission event", taxReturnSubmissions.size());
        taxReturnSubmissionRepository.saveAll(taxReturnSubmissions);
        if (!taxReturnTaxReturnSubmissionMap.isEmpty()) {
            confirmationService.enqueueErrorResolutionEmail(taxReturnTaxReturnSubmissionMap);
        }
    }
}
