package gov.irs.directfile.models;

import java.io.Serializable;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;

public record FactTypeWithItem(@JsonProperty("$type") String type, JsonNode item) implements Serializable {}
