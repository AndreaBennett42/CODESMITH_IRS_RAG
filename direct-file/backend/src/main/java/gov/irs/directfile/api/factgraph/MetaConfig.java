package gov.irs.directfile.api.factgraph;

public class MetaConfig implements gov.irs.factgraph.definitions.meta.MetaConfigTrait {

    public MetaConfig(String version) {
        this.version = version;
    }

    public String version;

    @Override
    public String version() {
        return version;
    }
}
