package gov.irs.directfile.api.config;

import java.util.List;

import org.springdoc.core.customizers.OpenApiCustomizer;
import org.springdoc.core.models.GroupedOpenApi;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableConfigurationProperties(DirectFileConfigurationProperties.class)
public class OpenApiConfiguration {
    private final String apiVersion;
    private final List<String> externalPaths =
            List.of("/taxreturns/**", "/users/**", "/state-api/state-profile", "/state-api/authorization-code");
    private final List<String> internalPaths = List.of("/state-api/state-exported-facts/**", "/state-api/status/**");
    private final List<String> demoPaths = List.of("/taxreturns-demo/**", "/debug/**", "/loaders/**");

    public OpenApiConfiguration(DirectFileConfigurationProperties configProps) {
        apiVersion = configProps.getApiVersion();
    }

    @Bean
    public GroupedOpenApi externalEndpointsOpenApiGroup() {
        return GroupedOpenApi.builder()
                .group("external")
                .displayName("external endpoints")
                .pathsToMatch(getPathsWithApiVersionPrefix(externalPaths))
                .build();
    }

    @Bean
    public GroupedOpenApi internalEndpointsOpenApiGroup() {
        return GroupedOpenApi.builder()
                .group("internal")
                .displayName("internal endpoints")
                .pathsToMatch(getPathsWithApiVersionPrefix(internalPaths))
                .addOpenApiCustomizer(removeSecuritySchemes())
                .build();
    }

    @Bean
    public GroupedOpenApi developmentOnlyOpenApiGroup() {
        return GroupedOpenApi.builder()
                .group("development")
                .displayName("development endpoints")
                .pathsToMatch(getPathsWithApiVersionPrefix(demoPaths))
                .build();
    }

    @Bean
    public OpenApiCustomizer removeSecuritySchemes() {
        return openApi -> {
            // Clear all security requirements
            openApi.getComponents().setSecuritySchemes(null);
            openApi.getSecurity().clear();
        };
    }

    private String[] getPathsWithApiVersionPrefix(List<String> paths) {
        return paths.stream().map(path -> "/" + apiVersion + path).toList().toArray(String[]::new);
    }
}
