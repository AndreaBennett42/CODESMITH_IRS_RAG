package gov.irs.boot.autoconfigure.openfeature;

import java.util.HashMap;
import java.util.Map;

import dev.openfeature.sdk.Client;
import dev.openfeature.sdk.FeatureProvider;
import dev.openfeature.sdk.OpenFeatureAPI;
import dev.openfeature.sdk.providers.memory.Flag;
import dev.openfeature.sdk.providers.memory.Flag.FlagBuilder;
import dev.openfeature.sdk.providers.memory.InMemoryProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

@AutoConfiguration
@EnableConfigurationProperties({FeatureFlagsConfigurationProperties.class, OpenFeatureConfigurationProperties.class})
public class OpenFeatureAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    Client featureFlagsClient(FeatureProvider provider) {
        OpenFeatureAPI api = OpenFeatureAPI.getInstance();
        api.setProviderAndWait(provider);

        return api.getClient();
    }

    @Bean
    @ConditionalOnMissingBean
    FeatureProvider yamlProvider(FeatureFlagsConfigurationProperties featureFlagsConfigurationProperties) {
        Map<String, Flag<?>> featureFlags = new HashMap<>();
        featureFlagsConfigurationProperties.featureFlags().entrySet().stream().forEach(featureFlag -> {
            FlagBuilder<Object> flagBuilder = Flag.builder();
            featureFlag.getValue().getVariants().entrySet().stream().forEach(variant -> {
                Object value = variant.getValue();
                if (value.toString().matches("true|false")) {
                    value = Boolean.parseBoolean(value.toString());
                }
                flagBuilder.variant(variant.getKey(), value);
            });
            flagBuilder.defaultVariant(featureFlag.getValue().getDefaultVariant());
            featureFlags.put(featureFlag.getKey(), flagBuilder.build());
        });

        return new InMemoryProvider(featureFlags);
    }

    @Bean
    @ConditionalOnMissingBean
    FeatureProvider s3Provider(FeatureFlagsConfigurationProperties featureFlagsConfigurationProperties) {
        Map<String, Flag<?>> featureFlags = new HashMap<>();
        // TODO

        return new InMemoryProvider(featureFlags);
    }
}
