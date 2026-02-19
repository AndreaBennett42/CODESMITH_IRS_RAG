package gov.irs.directfile.stateapi.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import reactor.core.publisher.Mono;

import gov.irs.directfile.dto.AuthCodeResponse;
import gov.irs.directfile.stateapi.model.AuthCodeRequest;
import gov.irs.directfile.stateapi.model.ExportResponse;

@Tag(name = "State-Tax-API", description = "Direct File State Tax APIs")
@SuppressWarnings("PMD.UnnecessaryModifier")
public interface StateApi {
    @Operation(
            summary = "Create authorization code",
            description =
                    "Create an authorization code for the given tax return UUID. This API will be called from Direct File service.")
    @ApiResponses({
        @ApiResponse(
                responseCode = "201",
                description = "Authorization code created",
                content =
                        @Content(
                                mediaType = "text/plain",
                                schema = @Schema(implementation = String.class),
                                examples =
                                        @ExampleObject(
                                                name = "authorization code example",
                                                value = "cd19876a-328c-4173-b4e6-59b55f4bb99e"))),
        @ApiResponse(
                responseCode = "500",
                description =
                        "Error occurred to create authorization code. For tax return not accepted, error message is E_TAX_RETURN_NOT_ACCEPTED; for tax return not found, error message is E_TAX_RETURN_NOT_FOUND",
                content = {
                    @Content(
                            mediaType = "text/plain",
                            examples = {
                                @ExampleObject(
                                        name = "Internal server error",
                                        value = "An internal server error occurred.")
                            })
                })
    })
    Mono<ResponseEntity<AuthCodeResponse>> createAuthorizationCode(
            @io.swagger.v3.oas.annotations.parameters.RequestBody(
                            description = "Request payload for creating the authorization code",
                            content =
                                    @Content(
                                            mediaType = "application/json",
                                            schema = @Schema(implementation = AuthCodeRequest.class),
                                            examples = @ExampleObject(value = AuthCodeRequest.docsExampleObject)))
                    @Valid
                    @RequestBody
                    AuthCodeRequest ac,
            HttpServletRequest request);

    @Operation(
            summary = "Export tax return",
            description =
                    "Export an encrypted tax return for a particular taxpayer identified by the authorization code passed in the JWT Bearer token. If the operation is successful, it will return a status of 200. In case of an error, it will also return a status of 200 along with an error code,"
                            + "e.g. {\"status\": \"error\", \"error\": \"E_CERTIFICATE_EXPIRED\"}. This API will be called from state-app.")
    @ApiResponse(
            responseCode = "200",
            description = "export taxpayer's federal tax return",
            content = {
                @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ExportResponse.class),
                        examples = @ExampleObject(value = ExportResponse.docsExampleObjectSuccess))
            })
    public Mono<ResponseEntity<ExportResponse>> exportReturn(
            @Parameter(
                            description = "Authorization header",
                            required = true,
                            examples = {@ExampleObject(name = "JWT Bearer token", value = "Bearer token")})
                    @RequestHeader("Authorization")
                    String authorizationHeader,
            HttpServletRequest request);
}
