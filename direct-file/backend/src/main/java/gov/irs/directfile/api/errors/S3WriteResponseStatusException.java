package gov.irs.directfile.api.errors;

import org.apache.commons.lang3.StringUtils;

import gov.irs.directfile.api.taxreturn.ApiErrorKeys;
import gov.irs.directfile.api.taxreturn.TaxReturnApi.S3WriteError;

public class S3WriteResponseStatusException extends ApiResponseStatusException {
    public static final String docsExampleObject =
            """
        {
            "status":"BAD_REQUEST",
            "message":"400 BAD_REQUEST \\"Unable to write XML for tax return to S3.\\"",
            "apiErrorKey":"externalServiceError",
            "body":{}
        }
            """;

    public S3WriteResponseStatusException(String message, Throwable cause) {
        super(
                S3WriteError.code,
                String.format("%s%s%s", S3WriteError.description, StringUtils.isBlank(message) ? "" : " ", message),
                ApiErrorKeys.S3_WRITE_ERROR,
                cause);
    }

    public S3WriteResponseStatusException(String message) {
        this(message, null);
    }

    public S3WriteResponseStatusException(Throwable cause) {
        this("", cause);
    }

    public S3WriteResponseStatusException() {
        this("", null);
    }
}
