package gov.irs.directfile.api.dataimport.model;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@RequiredArgsConstructor
public class WrappedPopulatedDataNode {

    private JsonNode payload;
    private String createdAt;
    private String state = WrappedPopulatedDataNodeState.INCOMPLETE.getState();

    public WrappedPopulatedDataNode(JsonNode payload, String createdAt, String state) {
        this.payload = payload;
        this.createdAt = createdAt;
        this.state = state;
    }
}
