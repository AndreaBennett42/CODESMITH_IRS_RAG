package gov.irs.directfile.api.config;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;

import gov.irs.directfile.api.authentication.FakePIIService;
import gov.irs.directfile.api.authentication.PIIAttribute;
import gov.irs.directfile.api.config.identity.IdentityAttributes;
import gov.irs.directfile.api.config.identity.IdentitySupplier;

@Configuration
@Profile(BeanProfiles.ENABLE_DEVELOPMENT_IDENTITY_SUPPLIER)
@ConfigurationPropertiesScan
@Slf4j
public class DevelopmentIdentitySupplier {
    static UUID internalId = UUID.fromString("11111111-1111-1111-1111-111111111111");
    static UUID externalId = UUID.fromString("00000000-0000-0000-0000-000000000000");
    private static final FakePIIService fakePIIService = new FakePIIService();
    private static final Set<PIIAttribute> piiAttributesToGenerate =
            Set.of(PIIAttribute.EMAILADDRESS, PIIAttribute.TIN);

    public record DevelopmentUserAttributes(String email, String tin) {}

    @ConfigurationProperties(prefix = "direct-file.dev-data.identity-supplier")
    public record DevelopmentUserProperties(Map<UUID, DevelopmentUserAttributes> userMap) {}

    private final Map<UUID, DevelopmentUserAttributes> externalIdToEmailMap;

    public DevelopmentIdentitySupplier(DevelopmentUserProperties developmentUserProperties) {
        log.info(
                "Running with development identity supplier.  Users will be loaded from \"direct-file.dev-data.identity-supplier.user-map\"");
        externalIdToEmailMap =
                developmentUserProperties != null ? developmentUserProperties.userMap() : new HashMap<>();
    }

    @Bean
    public IdentitySupplier getIdentitySupplierDevelopment() {
        return () -> {
            Map<PIIAttribute, String> piiAttributes =
                    fakePIIService.fetchAttributes(externalId, piiAttributesToGenerate);
            String email = piiAttributes.get(PIIAttribute.EMAILADDRESS);
            String tin = piiAttributes.get(PIIAttribute.TIN);
            return new IdentityAttributes(internalId, externalId, email, tin);
        };
    }
}
