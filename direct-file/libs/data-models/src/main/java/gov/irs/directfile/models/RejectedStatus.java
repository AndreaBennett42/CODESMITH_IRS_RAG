package gov.irs.directfile.models;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.EqualsAndHashCode;
import lombok.ToString;

@SuppressFBWarnings(
        value = {"NM_FIELD_NAMING_CONVENTION", "URF_UNREAD_PUBLIC_OR_PROTECTED_FIELD"},
        justification = "Initial Spotbugs Setup")
@EqualsAndHashCode
@ToString
public class RejectedStatus {
    public RejectedStatus(String meFErrorCode, String translationKey, String meFDescription) {
        MeFErrorCode = meFErrorCode;
        TranslationKey = translationKey;
        MeFDescription = meFDescription;
    }

    public String MeFErrorCode;
    public String TranslationKey;
    public String MeFDescription;
}
