package gov.irs.boot.validation.constraints.validators;

import jakarta.validation.ConstraintValidator;
import jakarta.validation.ConstraintValidatorContext;
import lombok.AllArgsConstructor;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import gov.irs.boot.validation.constraints.NotBlankForAws;

@AllArgsConstructor
@Component
public class NotBlankForAwsValidator implements ConstraintValidator<NotBlankForAws, String> {

    private final Environment env;

    @Override
    public boolean isValid(String value, ConstraintValidatorContext context) {
        if (env.matchesProfiles("aws")) {
            return StringUtils.hasText(value);
        }
        return true;
    }
}
