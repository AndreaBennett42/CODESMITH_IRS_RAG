package gov.irs.directfile.stateapi.exception;

import gov.irs.directfile.error.StateApiErrorCode;

public class StateNotExistException extends StateApiException {

    public StateNotExistException(StateApiErrorCode errorCode) {
        super(errorCode);
    }
}
