package gov.irs.directfile.api.loaders.domain;

import java.util.Optional;

public record PotentiallyIncompleteGraphGetResult(Optional value, Exception graphFetchError) {
    public boolean hasError() {
        return graphFetchError != null;
    }

    public String getErrorString() {
        if (!hasError()) {
            return "";
        }

        return graphFetchError.getMessage();
    }

    public Optional getValue() {
        if (hasError()) {
            throw new IllegalArgumentException("Graph result is an error");
        }
        return value;
    }
}
