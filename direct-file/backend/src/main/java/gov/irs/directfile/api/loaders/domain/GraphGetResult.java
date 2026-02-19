package gov.irs.directfile.api.loaders.domain;

public record GraphGetResult(String path, Object value, Exception graphFetchError) {
    public boolean hasError() {
        return graphFetchError != null;
    }

    public String getErrorString() {
        if (!hasError()) {
            return "";
        }

        return graphFetchError.getMessage();
    }

    public Object getValue() {
        if (hasError()) {
            throw new IllegalArgumentException("Graph result is an error");
        }
        return value;
    }
}
