package gov.irs.directfile.api.authentication;

import org.springframework.security.core.AuthenticationException;

public class EnrollmentWindowException extends AuthenticationException {
    public EnrollmentWindowException(String message) {
        super(message);
    }
}
