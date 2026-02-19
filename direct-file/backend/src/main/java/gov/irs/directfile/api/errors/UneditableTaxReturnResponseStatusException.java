package gov.irs.directfile.api.errors;

import org.apache.commons.lang3.StringUtils;

import gov.irs.directfile.api.taxreturn.ApiErrorKeys;
import gov.irs.directfile.api.taxreturn.TaxReturnApi.SubmittedTaxReturn;

public class UneditableTaxReturnResponseStatusException extends ApiResponseStatusException {
    public static final String docsExampleObject =
            """
        {
            "status":"CONFLICT",
            "message":"409 CONFLICT \\"Tax return has already been submitted, and is not editable.\\"",
            "apiErrorKey":"uneditableTaxReturn",
            "body":{}
        }
            """;

    public UneditableTaxReturnResponseStatusException(String message, Throwable cause) {
        super(
                SubmittedTaxReturn.code,
                String.format(
                        "%s%s%s", SubmittedTaxReturn.description, StringUtils.isBlank(message) ? "" : " ", message),
                ApiErrorKeys.UNEDITABLE_TAX_RETURN,
                cause);
    }

    public UneditableTaxReturnResponseStatusException(String message) {
        this(message, null);
    }

    public UneditableTaxReturnResponseStatusException(Throwable cause) {
        this("", cause);
    }

    public UneditableTaxReturnResponseStatusException() {
        this("", null);
    }
}
