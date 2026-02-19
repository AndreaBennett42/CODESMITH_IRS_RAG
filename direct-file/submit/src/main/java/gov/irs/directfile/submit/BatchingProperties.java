package gov.irs.directfile.submit;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public final class BatchingProperties {
    private int maxBatchSize;
    private long batchTimeoutMilliseconds;

    public BatchingProperties(int maxBatchSize, long batchTimeoutMilliseconds) {
        this.maxBatchSize = maxBatchSize;
        this.batchTimeoutMilliseconds = batchTimeoutMilliseconds;
    }

    @Override
    public String toString() {
        return "BatchingProperties{" + "maxBatchSize="
                + maxBatchSize + ", batchTimeoutMilliseconds="
                + batchTimeoutMilliseconds + '}';
    }
}
