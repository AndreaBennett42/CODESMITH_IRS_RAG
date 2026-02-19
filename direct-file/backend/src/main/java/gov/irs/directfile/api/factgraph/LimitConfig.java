package gov.irs.directfile.api.factgraph;

import gov.irs.factgraph.definitions.fact.CompNodeConfigTrait;
import gov.irs.factgraph.definitions.fact.LimitConfigTrait;
import gov.irs.factgraph.definitions.fact.LimitLevel;

public class LimitConfig implements LimitConfigTrait {

    private String operation;
    private LimitLevel level;
    private CompNodeConfigTrait node;

    public LimitConfig(final String operation, final LimitLevel level, final CompNodeConfigTrait node) {
        this.operation = operation;
        this.level = level;
        this.node = node;
    }

    @Override
    public String operation() {
        return operation;
    }

    @Override
    public LimitLevel level() {
        return level;
    }

    @Override
    public CompNodeConfigTrait node() {
        return node;
    }
}
