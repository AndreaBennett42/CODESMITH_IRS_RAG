package gov.irs.directfile.api.errors;

import org.apache.commons.lang3.StringUtils;

import gov.irs.directfile.api.taxreturn.ApiErrorKeys;
import gov.irs.directfile.api.taxreturn.TaxReturnApi.GenericResponseBadId;

public class TaxReturnNotFoundResponseStatusException extends ApiResponseStatusException {
    public static final String docsExampleObject =
            """
            {
                "status":"NOT_FOUND",
                "message":"404 NOT_FOUND \\"The user has no such tax return.\\"",
                "apiErrorKey":"generic.submissionError",
                "body":{}
            }
            """;

    public TaxReturnNotFoundResponseStatusException(String message, Throwable cause) {
        super(
                GenericResponseBadId.code,
                String.format(
                        "%s%s%s", GenericResponseBadId.description, StringUtils.isBlank(message) ? "" : " ", message),
                ApiErrorKeys.SUBMISSION_ERROR,
                cause);
    }

    public TaxReturnNotFoundResponseStatusException(String message) {
        this(message, null);
    }

    public TaxReturnNotFoundResponseStatusException(Throwable cause) {
        this("", cause);
    }

    public TaxReturnNotFoundResponseStatusException() {
        this("", null);
    }
}
