package gov.irs.directfile.stateapi.model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class StateAndAuthCode {

    private String authorizationCode;
    private String stateCode;
}
