package gov.irs.directfile.api.taxreturn.dto;

import java.util.*;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class TaxReturnSubmissionResponseBody {
    private UUID id;
    private Date createdAt;

    private UUID submitUserId;
    private String receiptId;
    private Date submissionReceivedAt;

    @JsonIgnore
    public static final String docsExampleObject =
            """
            {
               "id": "2d59a07d-57ef-4392-8196-48ac29dce023",
                "createdAt": "2023-10-26 15:04:47.197",
                "submitUserId": "6b1259fd-8cdb-4efe-bcc8-ad40e604c98b",
                "receiptId": "12345620230215000001",
                "submissionReceivedAt": "2023-10-26 16:01:34.221"
            }
            """;
}
