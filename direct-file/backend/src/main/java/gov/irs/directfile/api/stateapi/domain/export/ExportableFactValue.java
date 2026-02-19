package gov.irs.directfile.api.stateapi.domain.export;

import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonValue;
import scala.math.BigDecimal;

public class ExportableFactValue<T> implements ExportableData<T> {
    private final T value;

    public static ExportableFactValue<String> ofString(String value) {
        if (value != null && value.isBlank()) {
            return new ExportableFactValue<>(null);
        }

        return new ExportableFactValue<>(value);
    }

    public static ExportableFactValue<Boolean> ofBoolean(Boolean value) {
        return new ExportableFactValue<>(value);
    }

    public static ExportableFactValue<BigDecimal> ofBigDecimal(BigDecimal value) {
        return new ExportableFactValue<>(value);
    }

    private ExportableFactValue(T value) {
        this.value = value;
    }

    @Override
    @JsonValue
    public T getValue() {
        return value;
    }

    @Override
    public String toString() {
        return value.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ExportableFactValue<T> that = (ExportableFactValue<T>) o;
        return Objects.equals(value, that.value);
    }

    @Override
    public int hashCode() {
        return value == null ? 0 : value.hashCode();
    }
}
