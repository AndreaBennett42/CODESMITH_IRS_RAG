package gov.irs.directfile.submit.domain;

import java.util.Objects;

public record SubmissionBatch(long batchId, String path) {
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SubmissionBatch that = (SubmissionBatch) o;
        return Objects.equals(path, that.path);
    }

    @Override
    public int hashCode() {
        return Objects.hash(path);
    }
}
