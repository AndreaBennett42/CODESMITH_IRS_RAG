package gov.irs.boot.autoconfigure.boilerplate;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties("boilerplate-starter")
@AllArgsConstructor
@Getter
public class BoilerplateConfigurationProperties {

    @NotNull private final NestedClass1 nestedClass1;

    @NotNull private final NestedClass2 nestedClass2;

    @AllArgsConstructor
    @Getter
    public static class NestedClass1 {
        @NotBlank
        private final String property1;
    }

    @AllArgsConstructor
    @Getter
    public static class NestedClass2 {
        @NotBlank
        private final String property1;

        @NotBlank
        private final String property2;

        @NotBlank
        private final String property3;
    }
}
