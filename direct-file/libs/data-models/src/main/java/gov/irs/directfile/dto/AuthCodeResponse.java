package gov.irs.directfile.dto;

import java.util.UUID;

import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;

import gov.irs.directfile.error.StateApiErrorCode;

@Builder
@Getter
@EqualsAndHashCode
public class AuthCodeResponse {
    private UUID authCode;
    private StateApiErrorCode errorCode;
}
