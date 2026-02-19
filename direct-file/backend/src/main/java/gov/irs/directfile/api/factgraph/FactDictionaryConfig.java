package gov.irs.directfile.api.factgraph;

import scala.collection.Iterable;
import scala.jdk.CollectionConverters;

public class FactDictionaryConfig implements gov.irs.factgraph.definitions.FactDictionaryConfigTrait {

    public FactDictionaryConfig(
            java.lang.Iterable<gov.irs.factgraph.definitions.fact.FactConfigTrait> facts,
            gov.irs.factgraph.definitions.meta.MetaConfigTrait meta) {
        this.facts = facts;
        this.meta = meta;
    }

    public java.lang.Iterable<gov.irs.factgraph.definitions.fact.FactConfigTrait> facts;
    public gov.irs.factgraph.definitions.meta.MetaConfigTrait meta;

    @Override
    public Iterable<gov.irs.factgraph.definitions.fact.FactConfigTrait> facts() {
        return CollectionConverters.IterableHasAsScala(facts).asScala().toSeq();
    }

    @Override
    public gov.irs.factgraph.definitions.meta.MetaConfigTrait meta() {
        return meta;
    }
}
