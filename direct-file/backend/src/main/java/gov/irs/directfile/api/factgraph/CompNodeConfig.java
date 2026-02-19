package gov.irs.directfile.api.factgraph;

import scala.collection.Iterable;
import scala.jdk.CollectionConverters;

public class CompNodeConfig implements gov.irs.factgraph.definitions.fact.CompNodeConfigTrait {

    public CompNodeConfig(
            String typeName,
            java.lang.Iterable<gov.irs.factgraph.definitions.fact.CompNodeConfigTrait> children,
            java.lang.Iterable<gov.irs.factgraph.definitions.fact.OptionConfigTrait> options) {
        this.typeName = typeName;
        this.children = children;
        this.options = options;
    }

    public String typeName;
    public java.lang.Iterable<gov.irs.factgraph.definitions.fact.CompNodeConfigTrait> children;
    public java.lang.Iterable<gov.irs.factgraph.definitions.fact.OptionConfigTrait> options;

    @Override
    public String typeName() {
        return typeName;
    }

    @Override
    public Iterable<gov.irs.factgraph.definitions.fact.CompNodeConfigTrait> children() {
        return CollectionConverters.IterableHasAsScala(children).asScala().toSeq();
    }

    @Override
    public Iterable<gov.irs.factgraph.definitions.fact.OptionConfigTrait> options() {
        return CollectionConverters.IterableHasAsScala(options).asScala().toSeq();
    }
}
