package gov.irs.directfile.api.featureflags;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@AllArgsConstructor
@NoArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class FeatureFlags {
    public static final String docsExampleObject =
            """
        {2
          "openEnrollment": {
            "newUsersAllowed": true,
            "maxUsers": 200000000
          },
          "esignatureEnabled": false
        }
            """;

    private OpenEnrollment openEnrollment;

    private boolean esignatureEnabled = false;
}
