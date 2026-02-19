package gov.irs.directfile.stateapi.model;

import java.util.UUID;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@JsonInclude(JsonInclude.Include.NON_NULL)
@AllArgsConstructor
@NoArgsConstructor
@Data
public class AuthCodeRequest {

    @NotNull private UUID taxReturnUuid;

    private String tin;

    @NotNull private int taxYear;

    @NotBlank
    private String stateCode;

    @NotBlank
    private String submissionId;

    @JsonIgnore
    public static final String docsExampleObject =
            """
            {
                "taxReturnUuid": "ae019609-99e0-4ef5-85bb-ad90dc302e70",
                "tin": "123456789",
                "taxYear": 2022,
                "stateCode": "DC",
                "submissionId":"12345678901234567890"
            }
            """;
}
