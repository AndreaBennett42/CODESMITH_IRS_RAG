package gov.irs.directfile.api.errors;

import java.util.Map;

import org.springframework.http.HttpStatusCode;

import gov.irs.directfile.api.taxreturn.ApiErrorKeys;

public record ApiErrorResponse(
        HttpStatusCode status, String message, ApiErrorKeys apiErrorKey, Map<String, Object> body) {}
