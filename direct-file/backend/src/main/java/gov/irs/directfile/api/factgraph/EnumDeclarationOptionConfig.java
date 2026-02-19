package gov.irs.directfile.api.factgraph;

public class EnumDeclarationOptionConfig implements gov.irs.factgraph.definitions.meta.EnumDeclarationOptionsTrait {

    public EnumDeclarationOptionConfig(String value) {
        this.value = value;
    }

    public String value;

    @Override
    public String value() {
        return value;
    }
}
