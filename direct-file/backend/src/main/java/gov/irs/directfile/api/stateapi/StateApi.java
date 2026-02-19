package gov.irs.directfile.api.stateapi;

import java.util.UUID;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import gov.irs.directfile.api.stateapi.domain.CreateAuthorizationCodeRequest;
import gov.irs.directfile.api.stateapi.domain.CreateAuthorizationCodeResponse;
import gov.irs.directfile.api.stateapi.domain.GetStateExportedFactsResponse;
import gov.irs.directfile.api.stateapi.domain.GetStateProfileResponse;
import gov.irs.directfile.models.StateOrProvince;
import gov.irs.directfile.models.TaxReturnStatus;

@RequestMapping("${direct-file.api-version}" + StateApiController.BASE_URL)
@Tag(name = "stateapi", description = "The state API")
public interface StateApi {
    @Operation(summary = "Create authorization code", description = "Authorization code request successfully created")
    @PostMapping("authorization-code")
    @ResponseStatus(HttpStatus.ACCEPTED)
    ResponseEntity<CreateAuthorizationCodeResponse> createAuthorizationCode(
            @Valid @NotNull @RequestBody CreateAuthorizationCodeRequest createAuthorizationCodeRequest);

    @Operation(
            summary = "Get the state profile from the corresponding state code",
            description = "Retrieved client-facing state profile information")
    @GetMapping("state-profile")
    ResponseEntity<GetStateProfileResponse> getStateProfile(@RequestParam StateOrProvince stateCode);

    @Operation(
            summary = "Get the facts marked as exportable to state systems for a specific submission",
            description = "A json-parseable list of facts and their values")
    @GetMapping("state-exported-facts/{submissionId}")
    ResponseEntity<GetStateExportedFactsResponse> getStateExportedFacts(
            @PathVariable String submissionId,
            @RequestParam(required = true, value = "stateCode") StateOrProvince stateCode,
            @RequestParam(required = true, value = "accountId") String accountId);

    @Operation(
            summary =
                    "Get the status of a tax return. Endpoint is for internal communication between Direct File services only.")
    @GetMapping("/status/{taxFilingYear}/{taxReturnId}/{submissionId}")
    ResponseEntity<TaxReturnStatus> getTaxReturnStatus(
            @PathVariable("taxFilingYear") int taxFilingYear,
            @PathVariable("taxReturnId") UUID taxReturnId,
            @PathVariable("submissionId") String requestedSubmissionId);
}
