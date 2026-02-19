package gov.irs.directfile.api.taxreturn.dto;

import java.util.Map;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;

import gov.irs.directfile.models.FactTypeWithItem;

public record SignRequestBody(
        @NotNull(message = "No facts provided") Map<@NotEmpty String, @NotNull FactTypeWithItem> facts,
        @NotNull(message = "Missing intent statement") String intentStatement) {
    public static final String docsExampleObject =
            """
   {
       "intentStatement": "I agree to the terms.",
       "facts": {
           "/taxYear": {
               "$type": "gov.irs.factgraph.persisters.IntWrapper",
               "item": 2023
           }
       }
   }""";
}
