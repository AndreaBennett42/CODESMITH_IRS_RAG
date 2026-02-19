package gov.irs.directfile.api.stateapi.domain.export;

import java.util.HashMap;

import scala.math.BigDecimal;

import gov.irs.directfile.models.EvaluatedFactInfo;

import static gov.irs.directfile.api.stateapi.domain.export.ExportUtils.*;

public class ExportableCollectionItem extends HashMap<String, ExportableFactValue<?>>
        implements ExportableData<HashMap<String, ExportableFactValue<?>>> {
    @Override
    public HashMap<String, ExportableFactValue<?>> getValue() {
        return this;
    }

    public ExportableFactValue<?> put(String factKey, EvaluatedFactInfo evaluatedFactInfo) {
        if (isNull(evaluatedFactInfo)) {
            return this.put(factKey);
        } else if (isBoolean(evaluatedFactInfo)) {
            return this.put(factKey, (Boolean) evaluatedFactInfo.value());
        } else {
            return this.put(factKey, evaluatedFactInfo.value().toString());
        }
    }

    public ExportableFactValue<?> put(String factKey, String factValue) {
        return this.put(factKey, ExportableFactValue.ofString(factValue));
    }

    public ExportableFactValue<?> put(String factKey, Boolean factValue) {
        return this.put(factKey, ExportableFactValue.ofBoolean(factValue));
    }

    public ExportableFactValue<?> put(String factKey, BigDecimal factValue) {
        return this.put(factKey, ExportableFactValue.ofBigDecimal(factValue));
    }

    public ExportableFactValue<?> put(String factKey) {
        return this.put(factKey, ExportableFactValue.ofString(null));
    }

    public ExportableCollectionItem sanitized() {
        // return empty map if all values are null
        return this.values().stream().allMatch((efv) -> efv.getValue() == null) ? new ExportableCollectionItem() : this;
    }
}
