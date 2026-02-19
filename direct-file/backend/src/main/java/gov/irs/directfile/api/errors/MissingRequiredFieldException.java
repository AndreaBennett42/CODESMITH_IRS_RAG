package gov.irs.directfile.api.errors;

import lombok.Getter;

@Getter
public class MissingRequiredFieldException extends Exception {
    public MissingRequiredFieldException(String message) {
        super(message);
    }

    public MissingRequiredFieldException(String message, Throwable cause) {
        super(message, cause);
    }
}
