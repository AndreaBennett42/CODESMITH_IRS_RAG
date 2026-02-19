package gov.irs.directfile.models.message;

import com.fasterxml.jackson.annotation.JsonValue;
import lombok.Getter;
import lombok.NonNull;

/***
 * @deprecated Replaced by {@link gov.irs.directfile.models.LepLanguage
 *             LepLanguage}
 * @see gov.irs.directfile.models.LepLanguage
 */
@Getter
@SuppressWarnings("PMD.UnnecessaryFullyQualifiedName")
public enum SendEmailQueueLanguageCodeEnum {
    EN("en"),
    ES("es");

    @JsonValue
    final String languageCode;

    SendEmailQueueLanguageCodeEnum(String languageCode) {
        this.languageCode = languageCode;
    }

    public static SendEmailQueueLanguageCodeEnum valueOfIgnoreCase(@NonNull String value) {
        for (SendEmailQueueLanguageCodeEnum languageCode : SendEmailQueueLanguageCodeEnum.values()) {
            if (languageCode.name().equalsIgnoreCase(value)) return languageCode;
        }
        throw new IllegalArgumentException("no valid enum constant for the specified value");
    }

    @Override
    public String toString() {
        return name().toLowerCase();
    }
}
