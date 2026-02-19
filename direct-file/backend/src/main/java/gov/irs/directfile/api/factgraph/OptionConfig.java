package gov.irs.directfile.api.factgraph;

public class OptionConfig implements gov.irs.factgraph.definitions.fact.OptionConfigTrait {
    public OptionConfig(String name, String value) {
        this.name = name;
        this.value = value;
    }

    public String name;
    public String value;

    @Override
    public String name() {
        return name;
    }

    @Override
    public String value() {
        return value;
    }
}
