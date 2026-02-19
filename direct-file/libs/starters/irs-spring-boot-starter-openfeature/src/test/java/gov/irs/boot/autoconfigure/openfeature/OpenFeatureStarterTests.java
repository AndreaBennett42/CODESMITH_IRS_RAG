package gov.irs.boot.autoconfigure.openfeature;

import dev.openfeature.sdk.Client;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

class OpenFeatureStarterTests {

    @Test
    void contextLoads() {
        new ApplicationContextRunner()
                .withPropertyValues(
                        "openfeature-starter.s3-provider.environment-prefix=",
                        "openfeature-starter.s3-provider.bucket=test",
                        "openfeature-starter.s3-provider.expiration=1m",
                        "feature-flags.feature1.variants.on=true",
                        "feature-flags.feature1.variants.off=false",
                        "feature-flags.feature1.default-variant=on",
                        "feature-flags.feature2.variants.on=true",
                        "feature-flags.feature2.variants.off=false",
                        "feature-flags.feature2.default-variant=off")
                .withConfiguration(AutoConfigurations.of(OpenFeatureAutoConfiguration.class))
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).hasSingleBean(Client.class);
                    // verify that boolean values were mapped correctly
                    Client featureFlagsClient = context.getBean(Client.class);
                    assertThat(featureFlagsClient.getBooleanValue("feature1", Boolean.FALSE))
                            .isTrue();
                    assertThat(featureFlagsClient.getBooleanValue("feature2", Boolean.TRUE))
                            .isFalse();
                });
    }
}
