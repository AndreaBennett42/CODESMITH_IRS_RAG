package gov.irs.directfile.api.taxreturn.submissions;

import com.fasterxml.jackson.annotation.JsonValue;
import lombok.Getter;
import lombok.NonNull;

/***
 * @deprecated Replaced by {@link gov.irs.directfile.models.email.HtmlTemplate
 *             HtmlTemplate}
 * @see gov.irs.directfile.models.email.HtmlTemplate
 */
@Deprecated
@Getter
public enum SendEmailQueueStatusEnum {
    ACCEPTED("accepted"),
    REJECTED("rejected"),
    SUBMITTED("submitted");

    @JsonValue
    final String status;

    SendEmailQueueStatusEnum(String status) {
        this.status = status;
    }

    public static SendEmailQueueStatusEnum valueOfIgnoreCase(@NonNull String value) {
        for (SendEmailQueueStatusEnum status : values()) {
            if (status.name().equalsIgnoreCase(value)) return status;
        }
        throw new IllegalArgumentException("no valid enum constant for the specified value");
    }

    @Override
    public String toString() {
        return name().toLowerCase();
    }
}
