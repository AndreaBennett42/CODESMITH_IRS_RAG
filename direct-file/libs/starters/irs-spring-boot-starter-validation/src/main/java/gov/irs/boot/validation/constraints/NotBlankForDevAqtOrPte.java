package gov.irs.boot.validation.constraints;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import jakarta.validation.Constraint;
import jakarta.validation.Payload;

import gov.irs.boot.validation.constraints.validators.NotBlankForDevAqtOrPteValidator;

@Documented
@Constraint(validatedBy = NotBlankForDevAqtOrPteValidator.class)
@Target(value = ElementType.FIELD)
@Retention(value = RetentionPolicy.RUNTIME)
public @interface NotBlankForDevAqtOrPte {
    String message() default "{gov.irs.constraint.NotBlankForDevAqtPte.message}";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
