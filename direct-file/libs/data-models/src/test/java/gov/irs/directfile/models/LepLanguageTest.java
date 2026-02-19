package gov.irs.directfile.models;

import java.util.Map;

import com.fasterxml.jackson.databind.*;
import lombok.SneakyThrows;
import org.junit.jupiter.api.Test;

import gov.irs.factgraph.Graph;
import gov.irs.factgraph.monads.Result;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

// TODO restore commented out tests when implementing
// https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/7146
class LepLanguageTest {
    private static final ObjectMapper mapper = new ObjectMapper();
    private static final String DIRECT_FILE_LANGUAGE_PREFERENCE_PATH = "/directFileLanguagePreference";
    private static final String FACTGRAPH_WITH_LEP_LANGUAGE_PREFERENCE_FORMAT =
            """
                {
                    "facts": {
                        "/languagePreference": {
                            "$type": "gov.irs.factgraph.persisters.EnumWrapper",
                            "item": {
                                "value": ["%s"],
                                "enumOptionsPath": "/languageOptions"
                            }
                        }
                    }
                }
            """;
    private static final String FACTGRAPH_WITH_LEP_AND_DF_LANGUAGE_PREFERENCE_FORMAT =
            """
                {
                    "facts": {
                        "/languagePreference": {
                            "$type": "gov.irs.factgraph.persisters.EnumWrapper",
                            "item": {
                                "value": ["%s"],
                                "enumOptionsPath": "/languageOptions"
                            }
                        },
                        "/directFileLanguagePreference": {
                            "$type": "gov.irs.factgraph.persisters.StringWrapper",
                            "item": "%s"
                        }
                    }
                }
            """;
    private static final String FACTGRAPH_WITH_DF_LANGUAGE_PREFERENCE_FORMAT =
            """
                {
                    "facts": {
                        "/directFileLanguagePreference": {
                            "$type": "gov.irs.factgraph.persisters.StringWrapper",
                            "item": "%s"
                        }
                    }
                }
            """;
    private static final String FACTGRAPH_WITH_NO_LANGUAGE_PREFERENCE =
            """
                {
                    "facts": {
                        }
                    }
                }
            """;

    // TODO: Load a generated factgraph to catch when the /languagePreference JSON
    // structure changes

    // @Test
    // void FromFactsWithLepSpanishOnlySelected() {
    //     Map<String, FactTypeWithItem> facts =
    //             getFacts(String.format(FACTGRAPH_WITH_LEP_LANGUAGE_PREFERENCE_FORMAT, "spanish"));
    //     assertThat(LepLanguage.fromFacts(facts)).isEqualTo(LepLanguage.SPANISH);
    // }

    @Test
    void FromFactsWithDfSpanishOnlySelected() {
        Result<Object> result = Result.apply("es", true);
        Graph graph = mock(Graph.class);
        when(graph.get(eq(DIRECT_FILE_LANGUAGE_PREFERENCE_PATH))).thenReturn(result);

        assertThat(LepLanguage.fromFactGraph(graph)).isEqualTo(LepLanguage.SPANISH);
    }

    // @Test
    // void FromFactsWithInvalidLepLanguagePreferenceOnly() throws IllegalArgumentException {
    //     Map<String, FactTypeWithItem> facts =
    //             getFacts(String.format(FACTGRAPH_WITH_LEP_LANGUAGE_PREFERENCE_FORMAT, "unknown"));

    //     assertThatThrownBy(() -> LepLanguage.fromFacts(facts)).isInstanceOf(IllegalArgumentException.class);
    // }

    @Test
    void FromFactsWithInvalidDfLanguagePreferenceOnly() throws IllegalArgumentException {
        Result<Object> result = Result.apply("unknown", true);
        Graph graph = mock(Graph.class);
        when(graph.get(eq(DIRECT_FILE_LANGUAGE_PREFERENCE_PATH))).thenReturn(result);

        assertThatThrownBy(() -> LepLanguage.fromFactGraph(graph)).isInstanceOf(IllegalArgumentException.class);
    }

    // @Test
    // void FromFactsWithInvalidLepAndValidDfLanguagePreferenceOnly() {
    //     Map<String, FactTypeWithItem> facts =
    //             getFacts(String.format(FACTGRAPH_WITH_LEP_AND_DF_LANGUAGE_PREFERENCE_FORMAT, "unknown", "es"));
    //     assertThatThrownBy(() -> LepLanguage.fromFacts(facts)).isInstanceOf(IllegalArgumentException.class);
    // }

    // @Test
    // void FromFactsWithValidLepSpanishAndInvalidDfLanguagePreferenceOnly() throws IllegalArgumentException {
    //     Map<String, FactTypeWithItem> facts =
    //             getFacts(String.format(FACTGRAPH_WITH_LEP_AND_DF_LANGUAGE_PREFERENCE_FORMAT, "spanish", "unknown"));

    //     assertThat(LepLanguage.fromFacts(facts)).isEqualTo(LepLanguage.SPANISH);
    // }

    @Test
    void FromFactsWithNoLanguagePreference() {
        Map<String, FactTypeWithItem> facts = getFacts(FACTGRAPH_WITH_NO_LANGUAGE_PREFERENCE);
        // Defaults to English

        Result<Object> result = null;
        Graph graph = mock(Graph.class);
        when(graph.get(eq(DIRECT_FILE_LANGUAGE_PREFERENCE_PATH))).thenReturn(result);

        assertThat(LepLanguage.fromFactGraph(graph)).isEqualTo(LepLanguage.ENGLISH);
    }

    @Test
    void GetDefaultIfNotEnabled() {
        assertThat(LepLanguage.getDefaultIfNotEnabled(LepLanguage.SPANISH)).isEqualTo(LepLanguage.SPANISH);
        // LepLanguage.KOREAN has not been enabled, so should default to
        // LepLanguage.ENGLISH
        assertThat(LepLanguage.getDefaultIfNotEnabled(LepLanguage.KOREAN)).isEqualTo(LepLanguage.ENGLISH);
    }

    @Test
    void GetChineseTraditionalFromCode() {
        assertThat(LepLanguage.fromCode("zh-hant")).isEqualTo(LepLanguage.CHINESETRADITIONAL);
    }

    @Test
    void GetChineseSimplifiedFromCode() {
        assertThat(LepLanguage.fromCode("zh-hans")).isEqualTo(LepLanguage.CHINESESIMPLIFIED);
    }

    @Test
    void ToCodeFromChineseTraditional() {
        assertThat(LepLanguage.CHINESETRADITIONAL.toCode()).isEqualTo("zh-hant");
    }

    @Test
    void ToCodeFromGetChineseSimplified() {
        assertThat(LepLanguage.CHINESESIMPLIFIED.toCode()).isEqualTo("zh-hans");
    }

    @SneakyThrows
    private Map<String, FactTypeWithItem> getFacts(String json) {
        JsonNode tree = mapper.readTree(json);
        TestFactsWrapper testFactsWrapper = mapper.treeToValue(tree, TestFactsWrapper.class);
        return testFactsWrapper.getFacts();
    }
}
