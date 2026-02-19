package gov.irs.directfile.stateapi.exception;

import gov.irs.directfile.error.StateApiErrorCode;

public class StateApiExportedFactsDisabledException extends StateApiException {
    public StateApiExportedFactsDisabledException() {
        super(StateApiErrorCode.E_EXPORTED_FACTS_DISABLED);
    }

    @Override
    public String getMessage() {
        return getErrorCode().name() + "Encountered unexpected configuration mismatch - exported facts are disabled in"
                + " the backend api, but not in the state-api";
    }
}
