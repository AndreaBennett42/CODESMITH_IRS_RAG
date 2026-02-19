package gov.irs.directfile.status.domain;

import lombok.AllArgsConstructor;
import lombok.Getter;

import gov.irs.efile.ValidationErrorListType;

@Getter
@AllArgsConstructor
public class ValidationErrorGrpWrapper {
    private String ruleNum;
    private String severityCd;
    private String errorMessageTxt;

    public static ValidationErrorGrpWrapper fromMefValidationErrorGrp(
            ValidationErrorListType.ValidationErrorGrp validationErrorGrp) {
        return new ValidationErrorGrpWrapper(
                validationErrorGrp.getRuleNum(),
                validationErrorGrp.getSeverityCd(),
                validationErrorGrp.getErrorMessageTxt());
    }
}
