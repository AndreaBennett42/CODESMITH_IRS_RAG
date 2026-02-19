package gov.irs.directfile.stateapi.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ExportResponse {
    @NotNull private String status;

    private String taxReturn;
    private String error;

    @JsonIgnore
    public static final String docsExampleObjectSuccess =
            """
            {
                "status": "success",
                "taxReturn": "encoded-encrypted-data"
            }
            """;

    public static final String docsExampleObjectError =
            """
            {
                "status": "error",
                "error": "E_AUTHORIZATION_CODE_EXPIRED"
            }
            """;
}
