package gov.irs.directfile.api.stateapi.domain.export;

import gov.irs.directfile.models.EvaluatedFactInfo;

public class ExportUtils {
    public static String removeSlashes(String path) {
        return path.replace("/", "");
    }

    public static boolean isCollection(EvaluatedFactInfo evaluatedFactInfo) {
        return "gov.irs.factgraph.types.Collection".equals(evaluatedFactInfo.type());
    }

    public static boolean isBoolean(EvaluatedFactInfo evaluatedFactInfo) {
        return "java.lang.Boolean".equals(evaluatedFactInfo.type());
    }

    public static boolean isBigDecimal(EvaluatedFactInfo evaluatedFactInfo) {
        return "scala.math.BigDecimal".equals(evaluatedFactInfo.type());
    }

    public static boolean isNull(EvaluatedFactInfo evaluatedFactInfo) {
        return evaluatedFactInfo.type() == null && evaluatedFactInfo.value() == null;
    }
}
