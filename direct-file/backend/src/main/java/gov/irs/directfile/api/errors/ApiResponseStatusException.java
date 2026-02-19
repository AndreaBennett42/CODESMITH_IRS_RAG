package gov.irs.directfile.api.errors;

import java.util.Map;

import lombok.Getter;
import lombok.Setter;
import org.springframework.http.HttpStatusCode;
import org.springframework.lang.Nullable;
import org.springframework.web.server.ResponseStatusException;

import gov.irs.directfile.api.taxreturn.*;

@Getter
@Setter
public class ApiResponseStatusException extends ResponseStatusException {

    private ApiErrorKeys apiErrorKey;

    public static final String docsExampleObject =
            """
            {
                "status": "",
                "message": "",
                "apiErrorKey": "",
                "body": {}
            }
            """;

    public ApiResponseStatusException(
            HttpStatusCode status, @Nullable String reason, ApiErrorKeys apiErrorKey, @Nullable Throwable cause) {
        this(status, reason, apiErrorKey, Map.of(), cause);
    }

    public ApiResponseStatusException(
            HttpStatusCode status,
            @Nullable String reason,
            ApiErrorKeys apiErrorKey,
            Map<String, Object> apiErrorDetail,
            @Nullable Throwable cause) {
        super(status, reason, cause);
        this.apiErrorKey = apiErrorKey;

        if (apiErrorDetail != null && !apiErrorDetail.isEmpty()) {
            apiErrorDetail.entrySet().forEach(entry -> {
                this.getBody().setProperty(entry.getKey(), entry.getValue());
            });
        }
    }
}
