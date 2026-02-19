package gov.irs.directfile.api.audit;

import org.apache.commons.text.CaseUtils;

public enum AuditLogElement {
    CYBER_ONLY,
    DETAIL,
    EMAIL,
    EVENT_ERROR_MESSAGE,
    EVENT_ID,
    EVENT_STATUS,
    EVENT_TYPE,
    GOOGLE_ANALYTICS_ID,
    MEF_SUBMISSION_ID,
    XXX_CODE,
    REMOTE_ADDRESS,
    REQUEST_METHOD,
    REQUEST_URI,
    RESPONSE_STATUS_CODE,
    SADI_TID_HEADER,
    SADI_USER_UUID,
    STATE_ID,
    TAX_PERIOD,
    TAX_RETURN_ID,
    TIN_TYPE,
    TIMESTAMP,
    USER_TIN,
    USER_TIN_TYPE,
    USER_TYPE,
    DATA_IMPORT_BEHAVIOR;

    @Override
    public String toString() {
        return CaseUtils.toCamelCase(super.toString(), false, '_');
    }

    public enum DetailElement {
        STATE_ACCOUNT_ID,
        MESSAGE;

        @Override
        public String toString() {
            return CaseUtils.toCamelCase(super.toString(), false, '_');
        }
    }
}
