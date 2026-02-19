package gov.irs.directfile.api.pdf;

import java.util.List;

public record TableConfig(
        String rowsCollectionPath,
        int itemsToSkip,
        int rowsPerPage,
        List<PdfTableFactExpressionMapping> columns,
        List<PdfTableFactExpressionMapping> oncePerPageFields) {}
