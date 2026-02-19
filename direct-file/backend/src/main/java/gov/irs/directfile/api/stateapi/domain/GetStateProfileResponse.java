package gov.irs.directfile.api.stateapi.domain;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class GetStateProfileResponse {
    private StateProfile stateProfile;
    private String error;
}
