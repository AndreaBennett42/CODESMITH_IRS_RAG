package gov.irs.directfile.models;

import java.util.Map;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class TestFactsWrapper {
    private Map<String, FactTypeWithItem> facts;
}
