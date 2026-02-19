package gov.irs.directfile.api.factgraph;

import java.util.List;

import scala.Option;
import scala.collection.Iterable;
import scala.jdk.CollectionConverters;

public class WritableConfig implements gov.irs.factgraph.definitions.fact.WritableConfigTrait {

    public WritableConfig(
            String typeName,
            java.lang.Iterable<gov.irs.factgraph.definitions.fact.OptionConfigTrait> options,
            String collectionItemAlias,
            java.lang.Iterable<gov.irs.factgraph.definitions.fact.LimitConfigTrait> limits) {
        this.typeName = typeName;
        this.options = options;
        this.collectionItemAlias = collectionItemAlias;
        this.limits = limits;
    }

    public WritableConfig(String typeName) {
        this(typeName, List.of(), null, List.of());
    }

    public String typeName;
    public java.lang.Iterable<gov.irs.factgraph.definitions.fact.OptionConfigTrait> options;
    public String collectionItemAlias;
    public java.lang.Iterable<gov.irs.factgraph.definitions.fact.LimitConfigTrait> limits;

    @Override
    public String typeName() {
        return typeName;
    }

    @Override
    public Iterable<gov.irs.factgraph.definitions.fact.OptionConfigTrait> options() {
        return CollectionConverters.IterableHasAsScala(options).asScala().toSeq();
    }

    @Override
    public Option<String> collectionItemAlias() {
        return Option.apply(collectionItemAlias);
    }

    @Override
    public Iterable<gov.irs.factgraph.definitions.fact.LimitConfigTrait> limits() {
        return CollectionConverters.IterableHasAsScala(limits).asScala().toSeq();
    }
}
