package gov.irs.directfile.api.loaders.domain;

import gov.irs.factgraph.limits.LimitViolation;

public record LimitViolationInfo(String path, Object requestedValue, LimitViolation limitViolation) {}
