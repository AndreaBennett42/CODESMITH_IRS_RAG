package gov.irs.directfile.api.factgraph;

import scala.collection.Iterable;
import scala.jdk.CollectionConverters;

public class EnumDeclarationConfig implements gov.irs.factgraph.definitions.meta.EnumDeclarationTrait {

    public EnumDeclarationConfig(
            String id, java.lang.Iterable<gov.irs.factgraph.definitions.meta.EnumDeclarationOptionsTrait> options) {
        this.id = id;
        this.options = options;
    }

    public String id;
    public java.lang.Iterable<gov.irs.factgraph.definitions.meta.EnumDeclarationOptionsTrait> options;

    @Override
    public String id() {
        return id;
    }

    @Override
    public Iterable<gov.irs.factgraph.definitions.meta.EnumDeclarationOptionsTrait> options() {
        return CollectionConverters.IterableHasAsScala(options).asScala().toSeq();
    }
}
