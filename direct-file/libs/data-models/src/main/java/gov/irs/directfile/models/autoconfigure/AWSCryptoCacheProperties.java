package gov.irs.directfile.models.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Getter
@Setter
@ConfigurationProperties("aws.crypto-cache")
public class AWSCryptoCacheProperties {
    // these are mutable, to support them being:
    //   1) overridable without providing all properties
    //   2) provide default so apps including data-models don't have to provide a default if not using this
    private int messageUseLimit = 10;
    private int maxAgeSeconds = 60;
    private int maxItems = 10;
}
