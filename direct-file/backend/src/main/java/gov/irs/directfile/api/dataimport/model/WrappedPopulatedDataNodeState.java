package gov.irs.directfile.api.dataimport.model;

import lombok.Getter;

@Getter
public enum WrappedPopulatedDataNodeState {
    SUCCESS("success"),
    INCOMPLETE("incomplete");

    private final String state;

    WrappedPopulatedDataNodeState(String state) {
        this.state = state;
    }
}
