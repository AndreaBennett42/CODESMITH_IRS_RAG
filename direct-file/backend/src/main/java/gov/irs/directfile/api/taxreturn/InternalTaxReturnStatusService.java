package gov.irs.directfile.api.taxreturn;

import java.util.UUID;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.audit.AuditLogElement;
import gov.irs.directfile.api.audit.AuditService;
import gov.irs.directfile.api.io.documentstore.S3StorageService;
import gov.irs.directfile.api.taxreturn.dto.Status;
import gov.irs.directfile.api.taxreturn.models.SubmissionEvent;
import gov.irs.directfile.models.TaxReturnStatus;

@Service
@Slf4j
@AllArgsConstructor
public class InternalTaxReturnStatusService {
    private final TaxReturnService taxReturnService;
    private final S3StorageService s3StorageService;
    private final AuditService auditService;

    public TaxReturnStatus getTaxReturnStatusInternal(
            int taxFilingYear, UUID taxReturnId, String requestedSubmissionId) {
        auditService.addEventProperty(AuditLogElement.MEF_SUBMISSION_ID, requestedSubmissionId);
        auditService.addEventProperty(AuditLogElement.TAX_PERIOD, String.valueOf(taxFilingYear));
        MDC.put(AuditLogElement.TAX_RETURN_ID.toString(), taxReturnId.toString());

        try {
            SubmissionEvent submissionEvent =
                    taxReturnService.getLatestSubmissionEventByTaxReturnIdPreferringAcceptedSubmission(taxReturnId);
            String submissionId = submissionEvent.getSubmission().getSubmissionId();

            if (!requestedSubmissionId.equals(submissionId)) {
                log.warn(
                        "Using submission id {} instead of {} as the relevant submission Id for determining the status of tax return {}",
                        submissionId,
                        requestedSubmissionId,
                        taxReturnId);
            }

            Status status = submissionEvent.getStatus();
            String objectKey = generateSubmissionLocationObjectKey(taxFilingYear, taxReturnId, submissionId);
            boolean exists = s3StorageService.doesObjectAlreadyExist(objectKey);

            log.info("getTaxReturnStatus returns successfully, exists: {}", exists);
            return new TaxReturnStatus(status.toString(), exists);
        } catch (Exception e) {
            log.error(
                    "getTaxReturnStatus failed for taxFilingYear={}, taxReturnId={}, {}",
                    taxFilingYear,
                    taxReturnId,
                    e.getClass().getName());
            return new TaxReturnStatus(Status.Error.name(), false);
        }
    }

    private String generateSubmissionLocationObjectKey(int taxFilingYear, UUID taxReturnId, String submissionId) {
        return taxFilingYear + "/taxreturns/" + taxReturnId + "/submissions/" + submissionId + ".xml";
    }
}
