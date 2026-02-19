package gov.irs.directfile.api.errors;

import org.apache.commons.lang3.StringUtils;

import gov.irs.directfile.api.taxreturn.ApiErrorKeys;
import gov.irs.directfile.api.taxreturn.TaxReturnApi.FactsFailedToParseCorrectly;

public class FactGraphParseResponseStatusException extends ApiResponseStatusException {
    public static final String docsExampleObject =
            """
        {
            "status":"BAD_REQUEST",
            "message":"400 BAD_REQUEST \\"Could not parse facts for the provided fact graph.\\"",
            "apiErrorKey":"generic.submissionError",
            "body":{}
        }
            """;

    public FactGraphParseResponseStatusException(String message, Throwable cause) {
        super(
                FactsFailedToParseCorrectly.code,
                String.format(
                        "%s%s%s",
                        FactsFailedToParseCorrectly.description, StringUtils.isBlank(message) ? "" : " ", message),
                ApiErrorKeys.FACT_GRAPH_PARSE,
                cause);
    }

    public FactGraphParseResponseStatusException(String message) {
        this(message, null);
    }

    public FactGraphParseResponseStatusException(Throwable cause) {
        this("", cause);
    }

    public FactGraphParseResponseStatusException() {
        this("", null);
    }
}
