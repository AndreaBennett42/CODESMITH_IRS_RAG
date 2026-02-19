package gov.irs.directfile.api.taxreturn.dto;

import java.util.Date;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.*;

import gov.irs.directfile.models.RejectedStatus;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode
@ToString
public class StatusResponseBody {
    private Status status;
    private String translationKey;
    private List<RejectedStatus> rejectionCodes;
    private Date createdAt;

    @JsonIgnore
    public static final String docsExampleObject =
            """
                    {
                        "status": "Accepted",
                        "translationKey": "status.accepted"
                        "rejectionCodes": "[]"
                        "createdAt": "2023-10-26 15:04:47.197"
                    }
                    """;
}
