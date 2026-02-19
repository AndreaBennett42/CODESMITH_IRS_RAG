package gov.irs.directfile.api.taxreturn.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class XmlValidationResponseBody {

    @JsonProperty("isValid")
    private boolean isValid;
}
