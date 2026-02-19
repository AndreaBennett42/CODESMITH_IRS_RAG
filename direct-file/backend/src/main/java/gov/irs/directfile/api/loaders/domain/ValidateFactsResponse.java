package gov.irs.directfile.api.loaders.domain;

import java.util.Map;

import gov.irs.factgraph.limits.LimitViolation;

public record ValidateFactsResponse(Map<String, LimitViolation> errors, Map<String, Object> mess) {}
