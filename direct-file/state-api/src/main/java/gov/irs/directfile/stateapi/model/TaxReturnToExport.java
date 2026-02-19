package gov.irs.directfile.stateapi.model;

import java.util.Map;

import com.fasterxml.jackson.annotation.JsonInclude;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.USE_DEFAULTS)
public class TaxReturnToExport {
    @NotNull private String status;

    private String submissionId;
    private String xml;
    private Map<String, Object> directFileData;
}
