package gov.irs.directfile.api.taxreturn.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import gov.irs.directfile.models.Dispatch;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class XmlValidationRequestBody {
    private Dispatch dispatch;
}
