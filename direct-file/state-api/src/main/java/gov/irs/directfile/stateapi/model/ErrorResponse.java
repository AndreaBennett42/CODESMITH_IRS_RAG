package gov.irs.directfile.stateapi.model;

import lombok.Builder;
import lombok.Data;

import gov.irs.directfile.error.StateApiErrorCode;

@Data
@Builder
public class ErrorResponse {
    private StateApiErrorCode errorCode;
    private String errorMessage;
}
