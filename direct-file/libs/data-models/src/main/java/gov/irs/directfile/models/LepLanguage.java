package gov.irs.directfile.models;

import java.util.Locale;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import gov.irs.factgraph.Graph;
import gov.irs.factgraph.monads.Result;

/***
 * IRS LEP (Limited English Proficiency) Language Enumeration
 *
 */
@Slf4j
@SuppressWarnings("PMD.UnnecessaryFullyQualifiedName")
public enum LepLanguage {
    ENGLISH(Locale.ENGLISH),
    SPANISH("es"),
    KOREAN(Locale.KOREAN, false),
    VIETNAMESE("vi", false),
    RUSSIAN("ru", false),
    ARABIC("ar", false),
    HAITIAN("ht", false),
    TAGALOG("tl", false),
    PORTUGUESE("pt", false),
    POLISH("pl", false),
    FARSI("fa", false),
    FRENCH(Locale.FRENCH, false),
    JAPANESE(Locale.JAPANESE, false),
    GUJARATI("gu", false),
    PUNJABI("pa", false),
    KHMER("km", false),
    URDU("ur", false),
    BENGALI("bn", false),
    ITALIAN(Locale.ITALIAN, false),
    CHINESETRADITIONAL("zh-hant", false),
    CHINESESIMPLIFIED("zh-hans", false);

    private static final String FORMAT_LANGUAGE_NOT_FOUND_FOR_LANGUAGE_PREFERENCE_S =
            "Language not found for language preference: %s";
    private static final String PATH_LANGUAGE_PREFERENCE = "/languagePreference";
    private static final String PATH_DIRECT_FILE_LANGUAGE_PREFERENCE = "/directFileLanguagePreference";
    private static final LepLanguage DEFAULT_LANGUAGE = LepLanguage.ENGLISH;

    @Getter
    private Locale locale;

    @Getter
    private boolean enabled;

    LepLanguage(Locale locale, boolean enabled) {
        this.locale = locale;
        this.enabled = enabled;
    }

    LepLanguage(Locale locale) {
        this(locale, true);
    }

    LepLanguage(String languageCode, boolean enabled) {
        this.enabled = enabled;
        locale = Locale.forLanguageTag(languageCode);
    }

    LepLanguage(String languageCode) {
        this(languageCode, true);
    }

    @JsonValue
    public String toCode() {
        StringBuilder code = new StringBuilder().append(locale.getLanguage());
        if (StringUtils.isNotBlank(locale.getScript())) {
            code.append("-").append(locale.getScript());
        }
        return code.toString().toLowerCase();
    }

    public String getCode() {
        return toCode();
    }

    @JsonCreator
    public static LepLanguage fromCode(@NonNull String languageCode) {
        for (LepLanguage lepLanguage : LepLanguage.values()) {
            if (lepLanguage.toCode().equals(languageCode)) {
                return lepLanguage;
            }
        }
        throw new IllegalArgumentException(String.format("Language not found for code: %s", languageCode));
    }

    public static LepLanguage fromFactGraph(@NonNull Graph graph) {
        Result<Object> factGraphResult = graph.get(PATH_DIRECT_FILE_LANGUAGE_PREFERENCE);
        if (factGraphResult == null || !factGraphResult.hasValue()) {
            log.warn(
                    "Did not find language preference fact at path: {} or {}; defaulting to {}",
                    PATH_LANGUAGE_PREFERENCE,
                    PATH_DIRECT_FILE_LANGUAGE_PREFERENCE,
                    DEFAULT_LANGUAGE.name());
            return DEFAULT_LANGUAGE;
        }

        String languagePreference = factGraphResult.get().toString();
        try {
            return fromCode(languagePreference);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    String.format(FORMAT_LANGUAGE_NOT_FOUND_FOR_LANGUAGE_PREFERENCE_S, languagePreference), e);
        }
    }

    public static LepLanguage getDefaultIfNotEnabled(LepLanguage lepLanguage) {
        if (lepLanguage.isEnabled()) {
            return lepLanguage;
        }
        log.warn("{} is not enabled; defaulting to ENGLISH", lepLanguage.name());
        return DEFAULT_LANGUAGE;
    }
}
