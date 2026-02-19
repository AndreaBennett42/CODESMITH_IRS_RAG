package gov.irs.directfile.status.extension;

public final class BatchUtil {
    private BatchUtil() {
        // no instantiation allowed
    }

    public enum BatchType {
        INSERT,
        UPDATE,
        DELETE
    }

    public static String buildBatchMessage(int count, int batchSize, String entityName, BatchType type) {
        return String.format("Executing JDBC batch (%d / %d) - `%s#%s`", count, batchSize, entityName, type);
    }
}
