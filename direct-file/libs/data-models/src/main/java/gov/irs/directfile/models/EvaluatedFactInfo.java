package gov.irs.directfile.models;

import java.io.Serializable;

public record EvaluatedFactInfo(String type, Object value) implements Serializable {}
