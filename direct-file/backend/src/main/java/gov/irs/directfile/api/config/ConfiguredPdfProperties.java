package gov.irs.directfile.api.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class ConfiguredPdfProperties {
    private String name;
    private String year;
    private String languageCode;
    private String location;
    private String locationType;
    private String configurationLocation;
    private String configurationLocationType;
    private boolean cacheInMemory;
    private int[] pagesToInclude;
}
