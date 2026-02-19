package gov.irs.directfile.api.stateapi.domain;

import java.util.UUID;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Data;

@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
@Builder
public class CreateAuthorizationCodeResponse {
    private UUID authorizationCode;
    private String authorizationToken;
    private String error;
}
