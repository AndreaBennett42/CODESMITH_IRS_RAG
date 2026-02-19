package gov.irs.directfile.api.stateapi;

import java.util.Optional;
import java.util.UUID;

import com.fasterxml.jackson.core.JsonProcessingException;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import org.springframework.web.server.ResponseStatusException;

import gov.irs.factgraph.Graph;

import gov.irs.directfile.api.audit.AuditEventContextHolder;
import gov.irs.directfile.api.audit.AuditLogElement;
import gov.irs.directfile.api.audit.Auditable;
import gov.irs.directfile.api.config.StateApiEndpointProperties;
import gov.irs.directfile.api.config.StateApiFeatureFlagProperties;
import gov.irs.directfile.api.errors.InvalidDataException;
import gov.irs.directfile.api.errors.NonexistentDataException;
import gov.irs.directfile.api.events.EventId;
import gov.irs.directfile.api.events.UserType;
import gov.irs.directfile.api.loaders.errors.FactGraphSaveException;
import gov.irs.directfile.api.loaders.service.FactGraphService;
import gov.irs.directfile.api.stateapi.domain.*;
import gov.irs.directfile.api.taxreturn.InternalTaxReturnStatusService;
import gov.irs.directfile.api.taxreturn.TaxReturnService;
import gov.irs.directfile.api.taxreturn.TaxReturnSubmissionRepository;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;
import gov.irs.directfile.api.user.UserService;
import gov.irs.directfile.api.user.domain.UserInfo;
import gov.irs.directfile.dto.AuthCodeResponse;
import gov.irs.directfile.models.StateOrProvince;
import gov.irs.directfile.models.TaxReturnStatus;

@RestController
@Slf4j
@AllArgsConstructor
public class StateApiController implements StateApi {

    public static final String BASE_URL = "/state-api";
    private final UserService userService;
    private final TaxReturnService taxReturnService;
    private final StateApiService stateApiService;
    private final InternalTaxReturnStatusService internalTaxReturnStatusService;
    private final StateApiEndpointProperties stateApiEndpointProperties;
    private final StateApiFeatureFlagProperties stateApiFeatureFlagProperties;
    private final TaxReturnSubmissionRepository taxReturnSubmissionRepository;
    private final FactGraphService factGraphService;
    private final AuditEventContextHolder auditEventContextHolder;

    @Override
    @Auditable(event = EventId.CREATE_STATE_TAX_AUTHORIZATION_CODE)
    public ResponseEntity<CreateAuthorizationCodeResponse> createAuthorizationCode(
            CreateAuthorizationCodeRequest createAuthorizationCodeRequest) {

        log.info(
                "Received request to create authorization code for tax return {}, taxYear {}, api version {}",
                createAuthorizationCodeRequest.taxReturnUuid(),
                createAuthorizationCodeRequest.taxYear(),
                stateApiEndpointProperties.getVersion());

        UserInfo userInfo = userService.getCurrentUserInfo();

        Optional<TaxReturn> queriedTaxReturn =
                taxReturnService.findByIdAndUserId(createAuthorizationCodeRequest.taxReturnUuid(), userInfo.id());

        if (queriedTaxReturn.isEmpty()) {
            log.error(
                    "createAuthorizationCode failed: Unable to find tax return {} for user {}",
                    createAuthorizationCodeRequest.taxReturnUuid(),
                    userInfo.id());

            throw new NonexistentDataException("The user has no such tax return.");
        }

        var taxReturn = queriedTaxReturn.get();
        var calculatedFilingState = taxReturnService.getFilingStateOrProvince(taxReturn);

        if (calculatedFilingState.isEmpty()) {
            log.error(
                    "createAuthorizationCode failed: Unable to determine filing state from tax return {}",
                    taxReturn.getId());

            throw new InvalidDataException("A filing state cannot be determined from the tax return.");
        }

        try {
            var filingState = calculatedFilingState.get();
            MDC.put(AuditLogElement.STATE_ID.toString(), filingState.toString());

            Optional<TaxReturnSubmission> mostRecentSubmission =
                    taxReturnSubmissionRepository.findLatestTaxReturnSubmissionByTaxReturnId(taxReturn.getId());

            if (mostRecentSubmission.isEmpty()) {
                log.error("createAuthorizationCode failed: No submissions found for tax return {}", taxReturn.getId());
                throw new InvalidDataException("No Submissions found for the tax return.");
            }

            var responseBuilder = CreateAuthorizationCodeResponse.builder();
            if ("2".equals(stateApiEndpointProperties.getVersion())) {
                var stateApiRequest = new StateApiCreateAuthorizationTokenRequest(
                        createAuthorizationCodeRequest.taxReturnUuid(),
                        userInfo.tin(),
                        createAuthorizationCodeRequest.taxYear(),
                        filingState,
                        mostRecentSubmission.get().getSubmissionId());
                String authorizationToken = stateApiService.getAuthorizationToken(stateApiRequest);
                responseBuilder.authorizationToken(authorizationToken);
            } else {
                var stateApiRequest = new StateApiCreateAuthorizationCodeRequest(
                        createAuthorizationCodeRequest.taxReturnUuid(),
                        userInfo.tin(),
                        createAuthorizationCodeRequest.taxYear(),
                        filingState,
                        mostRecentSubmission.get().getSubmissionId());
                AuthCodeResponse response = stateApiService.getAuthorizationCode(stateApiRequest);
                UUID authorizationCode = response.getAuthCode();
                responseBuilder.authorizationCode(authorizationCode);
            }

            return new ResponseEntity<>(responseBuilder.build(), HttpStatus.OK);
        } catch (WebClientResponseException e) {
            log.error("Error getting authorization code {}", e.getMessage());
            throw e;
        }
    }

    @Override
    @Auditable(event = EventId.GET_STATE_PROFILE)
    public ResponseEntity<GetStateProfileResponse> getStateProfile(StateOrProvince stateCode) {
        MDC.put(AuditLogElement.STATE_ID.toString(), stateCode.name());

        // Calling this function to ensure that USER_TIN and USER_TIN_TYPE are included
        // in the audit log
        userService.getCurrentUserInfo();

        return new ResponseEntity<>(
                GetStateProfileResponse.builder()
                        .stateProfile(stateApiService.getStateProfile(stateCode))
                        .build(),
                HttpStatus.OK);
    }

    @Override
    @Auditable(event = EventId.GET_STATE_EXPORTED_FACTS_INTERNAL, type = UserType.SYS)
    public ResponseEntity<GetStateExportedFactsResponse> getStateExportedFacts(
            String submissionId, StateOrProvince stateCode, String accountId) {
        MDC.put(AuditLogElement.MEF_SUBMISSION_ID.toString(), submissionId);
        MDC.put(AuditLogElement.STATE_ID.toString(), stateCode.name());
        auditEventContextHolder.addValueToEventDetailMap(AuditLogElement.DetailElement.STATE_ACCOUNT_ID, accountId);
        log.info("Received request for exported facts with submissionID {}", submissionId);

        Optional<TaxReturnSubmission> trSubmission =
                taxReturnSubmissionRepository.findSubmissionBySubmissionId(submissionId);

        if (trSubmission.isEmpty()) {
            log.error("getStateExportedFacts failed: No submissions found for submissionId {}", submissionId);
            throw new ResponseStatusException(HttpStatus.NOT_FOUND);
        }

        var facts = trSubmission.get().getFacts();
        Graph graph = factGraphService.getGraph(facts);
        try {
            var factsExportedToState = stateApiService.getExportToStateFacts(graph);
            return new ResponseEntity<>(new GetStateExportedFactsResponse(factsExportedToState), HttpStatus.OK);
        } catch (JsonProcessingException | FactGraphSaveException e) {
            log.error("getStateExportedFacts failed to evaluate fact graph: {}", e.getMessage(), e);
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, e.getMessage(), e);
        } catch (Exception e) {
            log.error("getStateExportedFacts failed unexpectedly: {}", e.getMessage(), e);
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, e.getMessage(), e);
        }
    }

    @Override
    @Auditable(event = EventId.TAX_RETURN_GET_STATUS_INTERNAL, type = UserType.SYS)
    public ResponseEntity<TaxReturnStatus> getTaxReturnStatus(
            int taxFilingYear, UUID taxReturnId, String requestedSubmissionId) {
        TaxReturnStatus taxReturnStatus = internalTaxReturnStatusService.getTaxReturnStatusInternal(
                taxFilingYear, taxReturnId, requestedSubmissionId);
        return ResponseEntity.ok(taxReturnStatus);
    }
}
