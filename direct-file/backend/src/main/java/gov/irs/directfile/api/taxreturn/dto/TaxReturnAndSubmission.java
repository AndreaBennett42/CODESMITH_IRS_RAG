package gov.irs.directfile.api.taxreturn.dto;

import lombok.Getter;

import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;

@Getter
public class TaxReturnAndSubmission {
    private final TaxReturn taxReturn;
    private final TaxReturnSubmission taxReturnSubmission;

    public TaxReturnAndSubmission(TaxReturn taxReturn, TaxReturnSubmission taxReturnSubmission) {
        this.taxReturn = taxReturn;
        this.taxReturnSubmission = taxReturnSubmission;
    }
}
