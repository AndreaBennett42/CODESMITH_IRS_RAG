package gov.irs.directfile.submit.actions.exception;

import java.util.List;

import lombok.Getter;

import gov.irs.directfile.submit.actions.ActionException;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.domain.UserContextData;

@Getter
public class BundleArchiveActionException extends ActionException {
    private final List<UserContextData> userContextDataList;
    private SubmissionBatch batch;

    public BundleArchiveActionException(List<UserContextData> userContextDataList, SubmissionBatch batch, Throwable e) {
        super(e);
        this.userContextDataList = userContextDataList;
        this.batch = batch;
    }

    public String userContextDataTaxReturnIdsToString() {
        return "UserContextData{" + "TaxReturnIds="
                + getUserContextDataList().stream()
                        .map(UserContextData::getTaxReturnId)
                        .toList() + "}";
    }
}
