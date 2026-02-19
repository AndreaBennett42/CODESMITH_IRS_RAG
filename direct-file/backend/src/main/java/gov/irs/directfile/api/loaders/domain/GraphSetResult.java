package gov.irs.directfile.api.loaders.domain;

import java.util.List;

import gov.irs.factgraph.Graph;

public record GraphSetResult(Graph graph, List<LimitViolationInfo> limitViolationInfos) {
    public GraphSetResult {
        limitViolationInfos = List.copyOf(limitViolationInfos);
    }

    public boolean hasViolations() {
        return limitViolationInfos.size() > 0;
    }
}
