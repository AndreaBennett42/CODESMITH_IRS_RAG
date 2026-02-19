package gov.irs.directfile.stateapi.exception;

import lombok.Getter;

import gov.irs.directfile.error.StateApiErrorCode;

@Getter
public class StateApiException extends RuntimeException {

    private final StateApiErrorCode errorCode;

    public StateApiException(StateApiErrorCode errorCode) {
        super(errorCode.name());
        this.errorCode = errorCode;
    }

    public StateApiException(StateApiErrorCode errorCode, Throwable e) {
        super(errorCode.name(), e);
        this.errorCode = errorCode;
    }
}
