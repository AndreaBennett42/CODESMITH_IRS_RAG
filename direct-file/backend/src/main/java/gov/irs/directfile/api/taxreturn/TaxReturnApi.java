package gov.irs.directfile.api.taxreturn;

import java.util.List;
import java.util.UUID;

import com.fasterxml.jackson.databind.JsonNode;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.headers.Header;
import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import gov.irs.directfile.api.dataimport.model.PopulatedData;
import gov.irs.directfile.api.dataimport.model.WrappedPopulatedData;
import gov.irs.directfile.api.errors.ApiResponseStatusException;
import gov.irs.directfile.api.errors.FactGraphParseResponseStatusException;
import gov.irs.directfile.api.errors.TaxReturnNotFoundResponseStatusException;
import gov.irs.directfile.api.errors.UneditableTaxReturnResponseStatusException;
import gov.irs.directfile.api.taxreturn.dto.CreateRequestBody;
import gov.irs.directfile.api.taxreturn.dto.ResponseBody;
import gov.irs.directfile.api.taxreturn.dto.SignRequestBody;
import gov.irs.directfile.api.taxreturn.dto.StatusResponseBody;
import gov.irs.directfile.api.taxreturn.dto.SubmitRequestBody;
import gov.irs.directfile.api.taxreturn.dto.UpdateRequestBody;

@RequestMapping("${direct-file.api-version}" + TaxReturnController.baseUrl)
@Validated
@SuppressWarnings(value = {"PMD.AvoidDuplicateLiterals", "PMD.SignatureDeclareThrowsException"})
@Tag(name = "taxreturns", description = "The tax return API")
public interface TaxReturnApi {

    @Operation(summary = "List tax returns", description = "List all of the user's tax returns")
    @ApiResponses(
            value = {
                @ApiResponse(
                        responseCode = TaxReturnController.GenericResponseSuccess.codeString,
                        description = TaxReturnController.GenericResponseSuccess.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    array =
                                            @ArraySchema(
                                                    schema = @Schema(implementation = ResponseBody.class),
                                                    uniqueItems = true),
                                    examples = @ExampleObject(value = ResponseBody.docsExampleList))
                        }),
            })
    @GetMapping(produces = "application/json")
    ResponseEntity<List<ResponseBody>> getAllByUserId();

    @Operation(summary = "Find a tax return", description = "Find a tax return by its ID")
    @ApiResponses(
            value = {
                @ApiResponse(
                        responseCode = TaxReturnController.GenericResponseSuccess.codeString,
                        description = TaxReturnController.GenericResponseSuccess.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ResponseBody.class),
                                    examples = @ExampleObject(value = ResponseBody.docsExampleObject))
                        }),
                @ApiResponse(
                        responseCode = TaxReturnController.GenericResponseBadId.codeString,
                        description = TaxReturnController.GenericResponseBadId.description,
                        content = {@Content(schema = @Schema())})
            })
    @GetMapping(path = "/{id}", produces = "application/json")
    ResponseEntity<ResponseBody> getById(@Parameter(description = "Tax return ID") @PathVariable UUID id);

    @Operation(
            summary = "Get imported data for a tax return",
            description = "Get imported data for a tax return by its ID")
    @ApiResponses(
            value = {
                @ApiResponse(
                        responseCode = TaxReturnController.GenericResponseSuccess.codeString,
                        description = TaxReturnController.GenericResponseSuccess.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(type = "array", implementation = JsonNode.class),
                                    examples = @ExampleObject(value = PopulatedData.docsExampleObject))
                        }),
                @ApiResponse(
                        responseCode = TaxReturnController.GenericResponseBadId.codeString,
                        description = TaxReturnController.GenericResponseBadId.description,
                        content = {@Content(schema = @Schema())})
            })
    @GetMapping(path = "/{id}/populate", produces = "application/json")
    WrappedPopulatedData getPopulatedData(@Parameter(description = "Tax return ID") @PathVariable UUID id);

    @Operation(summary = "Create a tax return", description = "Create a new tax return")
    @ApiResponses(
            value = {
                @ApiResponse(
                        responseCode = CreateResponseSuccess.codeString,
                        description = CreateResponseSuccess.description,
                        headers = {
                            @Header(
                                    name = "Location",
                                    description = "Tax return's URI",
                                    schema = @Schema(type = "string"))
                        },
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ResponseBody.class),
                                    examples = @ExampleObject(value = ResponseBody.docsExampleObject))
                        }),
                @ApiResponse(
                        responseCode = GenericResponseBadData.codeString,
                        description = GenericResponseBadData.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ApiResponseStatusException.class)),
                            @Content(
                                    mediaType = "*/*",
                                    schema = @Schema(implementation = ResponseStatusException.class),
                                    examples =
                                            @ExampleObject(
                                                    value =
                                                            """
                                                {
                                                    "timestamp":"2024-04-10T19:38:53.343+00:00",
                                                    "status":400,
                                                    "error":"Bad Request",
                                                    "message":"Invalid request data.",
                                                    "path":"/df/file/api/v1/taxreturns"
                                                }
                                                    """))
                        }),
                @ApiResponse(
                        responseCode = CreateResponseBadState.codeString,
                        description = CreateResponseBadState.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ResponseStatusException.class),
                                    examples =
                                            @ExampleObject(
                                                    value =
                                                            """
                                                {
                                                    "timestamp":"2024-04-10T19:31:14.739+00:00",
                                                    "status":409,
                                                    "error":"Conflict",
                                                    "message":"The user already has a tax return for that tax year.",
                                                    "path":"/df/file/api/v1/taxreturns"
                                                }
                                                    """))
                        }),
                @ApiResponse(
                        responseCode = "500",
                        description = "Internal Server Error",
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ResponseStatusException.class),
                                    examples =
                                            @ExampleObject(
                                                    value =
                                                            """
                                                      {
                                                          "timestamp":"2024-04-10T16:24:57.808+00:00",
                                                          "status":500,
                                                          "error":"Internal Server Error",
                                                          "message":"java.lang.reflect.InvocationTargetException",
                                                          "path":"/df/file/api/v1/taxreturns/5c75bf5b-401a-40ec-ab6f-be1c6a299162"
                                                      }
                                            """)),
                        }),
            })
    @PostMapping(consumes = "application/json")
    ResponseEntity<ResponseBody> create(
            @io.swagger.v3.oas.annotations.parameters.RequestBody(
                            content = {
                                @Content(
                                        schema = @Schema(implementation = CreateRequestBody.class),
                                        examples = @ExampleObject(value = CreateRequestBody.docsExampleObject))
                            })
                    @Valid
                    @RequestBody
                    CreateRequestBody body,
            HttpServletRequest request);

    @Operation(
            summary = "Update a tax return",
            description = "Update an existing tax return, overwriting the existing data")
    @ApiResponses(
            value = {
                @ApiResponse(
                        responseCode = UpdateResponseSuccess.codeString,
                        description = UpdateResponseSuccess.description,
                        headers = {
                            @Header(
                                    name = "Location",
                                    description = "Tax return's URI",
                                    schema = @Schema(type = "string"))
                        },
                        content = {@Content(schema = @Schema())}),
                @ApiResponse(
                        responseCode = GenericResponseBadData.codeString,
                        description = GenericResponseBadData.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ResponseStatusException.class),
                                    examples =
                                            @ExampleObject(
                                                    value =
                                                            """
                                                {
                                                    "timestamp":"2024-04-10T16:01:54.163+00:00",
                                                    "status":400,
                                                    "error":"Bad Request",
                                                    "message":"Invalid request data.",
                                                    "path":"/df/file/api/v1/taxreturns/dedbffab-3b04-41b3-b8d2-f50c1aa7f19d"
                                                }
                                                    """))
                        }),
                @ApiResponse(
                        responseCode = GenericResponseBadId.codeString,
                        description = GenericResponseBadId.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ApiResponseStatusException.class),
                                    examples =
                                            @ExampleObject(
                                                    value =
                                                            TaxReturnNotFoundResponseStatusException
                                                                    .docsExampleObject)),
                        }),
                @ApiResponse(
                        responseCode = ModifyResponseBadState.codeString,
                        description = ModifyResponseBadState.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ApiResponseStatusException.class),
                                    examples =
                                            @ExampleObject(
                                                    value =
                                                            UneditableTaxReturnResponseStatusException
                                                                    .docsExampleObject)),
                            @Content(
                                    mediaType = "*/*",
                                    schema = @Schema(implementation = ResponseStatusException.class),
                                    examples =
                                            @ExampleObject(
                                                    value =
                                                            """
                                                        {
                                                            "timestamp":"2024-04-10T15:55:18.614+00:00",
                                                            "status":409,
                                                            "error":"Conflict",
                                                            "message":"The tax return has already been dispatched for electronic filing.",
                                                            "path":"/df/file/api/v1/taxreturns/525fc5ee-1b6a-47ad-9ab1-5bfb3edef9df"
                                                        }
                                                            """))
                        }),
                @ApiResponse(
                        responseCode = "500",
                        description = "Internal Server Error",
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ResponseStatusException.class),
                                    examples =
                                            @ExampleObject(
                                                    value =
                                                            """
                                                      {
                                                          "timestamp":"2024-04-10T16:24:57.808+00:00",
                                                          "status":500,
                                                          "error":"Internal Server Error",
                                                          "message":"java.lang.reflect.InvocationTargetException",
                                                          "path":"/df/file/api/v1/taxreturns/5c75bf5b-401a-40ec-ab6f-be1c6a299162"
                                                      }
                                            """)),
                        }),
            })
    @PostMapping(path = "/{id}", consumes = "application/json")
    ResponseEntity<Void> update(
            @PathVariable UUID id,
            @io.swagger.v3.oas.annotations.parameters.RequestBody(
                            content = {
                                @Content(
                                        schema = @Schema(implementation = UpdateRequestBody.class),
                                        examples = @ExampleObject(value = UpdateRequestBody.docsExampleObject))
                            })
                    @Valid
                    @RequestBody
                    UpdateRequestBody body,
            HttpServletRequest request);

    @Operation(summary = "Submit a tax return", description = "Submit a tax return for electronic filing")
    @ApiResponses(
            value = {
                @ApiResponse(
                        responseCode = SubmitResponseSuccess.codeString,
                        description = SubmitResponseSuccess.description,
                        content = {
                            @Content(
                                    mediaType = "text/plain",
                                    schema = @Schema(implementation = ResponseEntity.class),
                                    examples =
                                            @ExampleObject(
                                                    value =
                                                            "Tax return $TAX_RETURN_ID was dispatched to the electronic filing queue by user $USER_ID at $TAX_RETURN_MOST_RECENT_SUBMIT_TIME"))
                        }),
                @ApiResponse(
                        responseCode = GenericResponseBadData.codeString,
                        description = GenericResponseBadData.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ApiResponseStatusException.class),
                                    examples =
                                            @ExampleObject(
                                                    value = FactGraphParseResponseStatusException.docsExampleObject))
                        }),
                @ApiResponse(
                        responseCode = GenericResponseBadId.codeString,
                        description = GenericResponseBadId.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ApiResponseStatusException.class),
                                    examples =
                                            @ExampleObject(
                                                    value = TaxReturnNotFoundResponseStatusException.docsExampleObject))
                        }),
                @ApiResponse(
                        responseCode = ModifyResponseBadState.codeString,
                        description = ModifyResponseBadState.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ApiResponseStatusException.class),
                                    examples =
                                            @ExampleObject(
                                                    value =
                                                            UneditableTaxReturnResponseStatusException
                                                                    .docsExampleObject))
                        }),
                @ApiResponse(
                        responseCode = "500",
                        description = ModifyResponseBadState.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = ApiResponseStatusException.class),
                                    examples = @ExampleObject(value = ApiResponseStatusException.docsExampleObject)),
                        }),
            })
    @PostMapping(path = "/{id}/submit", produces = "text/plain")
    ResponseEntity<String> submit(
            @Parameter(description = "Tax return ID") @PathVariable UUID id,
            @io.swagger.v3.oas.annotations.parameters.RequestBody(
                            content = {
                                @Content(
                                        schema = @Schema(implementation = SubmitRequestBody.class),
                                        examples = @ExampleObject(value = SubmitRequestBody.docsExampleObject))
                            })
                    @Valid
                    @RequestBody
                    SubmitRequestBody body,
            HttpServletRequest request);

    @Operation(
            summary = "Accept a signed request",
            description = "Accepts a signed request prior to submitting the tax return")
    @ApiResponses(
            value = {
                @ApiResponse(
                        responseCode = SignedResponseSuccess.codeString,
                        description = SignedResponseSuccess.description,
                        content = {
                            @Content(mediaType = "text/plain", schema = @Schema(implementation = ResponseEntity.class)),
                        }),
                @ApiResponse(
                        responseCode = SignedResponseSigningDisable.codeString,
                        description = SignedResponseSuccess.description,
                        content = {
                            @Content(mediaType = "text/plain", schema = @Schema(implementation = ResponseEntity.class)),
                        })
            })
    @PostMapping(path = "/{id}/sign", consumes = "application/json")
    ResponseEntity<String> sign(
            @PathVariable UUID id,
            @io.swagger.v3.oas.annotations.parameters.RequestBody(
                            content = {
                                @Content(
                                        schema = @Schema(implementation = SignRequestBody.class),
                                        examples = @ExampleObject(value = SignRequestBody.docsExampleObject))
                            })
                    @Valid
                    @RequestBody
                    SignRequestBody body,
            HttpServletRequest request)
            throws Exception;

    @Operation(
            summary = "Check status of a tax return",
            description = "Check the status of a tax return filed with MeF")
    @ApiResponses(
            value = {
                @ApiResponse(
                        responseCode = GenericResponseSuccess.codeString,
                        description = GenericResponseSuccess.description,
                        content = {
                            @Content(
                                    mediaType = "application/json",
                                    schema = @Schema(implementation = StatusResponseBody.class),
                                    examples = @ExampleObject(value = StatusResponseBody.docsExampleObject))
                        }),
                @ApiResponse(
                        responseCode = GetStatusResponseBadId.codeString,
                        description = GetStatusResponseBadId.description,
                        content = {@Content(schema = @Schema())}),
                @ApiResponse(
                        responseCode = GetStatusBadState.codeString,
                        description = GetStatusBadState.description,
                        content = {@Content(schema = @Schema())})
            })
    @GetMapping(path = "/{id}/status", produces = "application/json")
    StatusResponseBody status(@Parameter(description = "Tax return ID") @PathVariable UUID id);

    @Operation(
            summary = "Get a PDF representing the current state of the tax return",
            description = "Generate a PDF that has all of the user information currently on the fact graph")
    @ApiResponses(
            value = {
                @ApiResponse(
                        responseCode = GenericResponseSuccess.codeString,
                        description = GenericResponseSuccess.description,
                        content = {@Content(mediaType = "application/pdf")}),
                @ApiResponse(
                        responseCode = GenericResponseBadId.codeString,
                        description = GenericResponseBadId.description,
                        content = {@Content(schema = @Schema())}),
                @ApiResponse(
                        responseCode = GetPdfResponseFailedToCreate.codeString,
                        description = GetPdfResponseFailedToCreate.description,
                        content = {@Content(schema = @Schema())})
            })
    @PostMapping(path = "/{id}/pdf/{languageCode}", produces = MediaType.APPLICATION_PDF_VALUE)
    ResponseEntity<InputStreamResource> pdf(
            @Parameter(description = "Tax return ID") @PathVariable UUID id,
            @Parameter(description = "Requested language") @PathVariable String languageCode);

    class GenericResponseSuccess {
        public static final String codeString = "200";
        public static final String description = "Successful operation.";
    }

    class GenericResponseBadId {
        public static final String codeString = "404";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "The user has no such tax return.";
    }

    class GenericResponseBadData {
        public static final String codeString = "400";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "Invalid request data.";
    }

    class CreateResponseSuccess {
        public static final String codeString = "201";
        public static final String description = "Created tax return.";
    }

    class CreateResponseBadState {
        public static final String codeString = "409";
        public static final HttpStatusCode code = HttpStatus.CONFLICT;
        public static final String description = "The user already has a tax return for that tax year.";
    }

    class DeleteResponseSuccess {
        public static final String codeString = "201";
        public static final String description = "Deleted tax return.";
    }

    class DeleteResponseBadState {
        public static final String codeString = "409";
        public static final HttpStatusCode code = HttpStatus.CONFLICT;
        public static final String description = "The user cannot delete their tax return for that tax year.";
    }

    class UpdateResponseSuccess {
        public static final String codeString = "204";
        public static final String description = "Updated tax return.";
    }

    class SubmitResponseSuccess {
        public static final String codeString = "202";
        public static final String description = "The submission was accepted.";
    }

    class SignedResponseSuccess {
        public static final String codeString = "202";
        public static final String description = "The signed request was accepted.";
    }

    class SignedResponseSigningDisable {
        public static final String codeString = "404";
        public static final String description = "The signed request was accepted.";
    }

    class ModifyResponseBadState {
        public static final String codeString = "409";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "The tax return has already been dispatched for electronic filing.";
    }

    class GetStatusBadState {
        public static final String codeString = "503";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "Could not reach the status service.";
    }

    class SubmitServiceUnavailable {
        public static final String codeString = "503";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "Could not reach the submit service.";
    }

    class GetStatusResponseBadId {
        public static final String codeString = "404";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description =
                "Could not find a submission ID for the requested return.  It may not have been processed yet.";
    }

    class InvalidIpAddress {
        public static final String codeString = "400";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "Unable to complete the requested action.";
    }

    class GetPdfResponseFailedToCreate {
        public static final String codeString = "500";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "Could not create the PDF.";
    }

    class EmailMismatch {
        public static final String codeString = "400";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "Email Validation failed. Tax Return Email must match ID.me.";
    }

    class TinMismatch {
        public static final String codeString = "400";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "TIN Validation failed. Tax Return TIN(s) must match ID.me.";
    }

    class SubmissionBlockingFacts {
        public static final String codeString = "400";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "Submission blocking facts are true for tax return.";
    }

    class SubmittedTaxReturn {
        public static final String codeString = "409";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "Tax return has already been submitted, and is not editable.";
    }

    class FactsFailedToParseCorrectly {
        public static final String codeString = "400";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "Could not parse facts for the provided fact graph.";
    }

    class BadSecurityStateOnCreate {
        public static final String codeString = "400";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description =
                "Creation failed because your tax return is missing internal data. You can't file your taxes with Direct File.";
    }

    class BadSecurityStateOnSubmission {
        public static final String codeString = "400";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description =
                "Submission failed because of missing internal data. Try submitting using with different browser or device.";
    }

    class S3WriteError {
        public static final String codeString = "400";
        public static final HttpStatusCode code = HttpStatus.valueOf(Integer.parseInt(codeString));
        public static final String description = "Unable to write XML for tax return to S3.";
    }
}
