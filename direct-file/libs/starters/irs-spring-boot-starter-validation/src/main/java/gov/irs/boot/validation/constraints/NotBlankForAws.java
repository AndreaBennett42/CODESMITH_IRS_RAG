package gov.irs.boot.validation.constraints;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import jakarta.validation.Constraint;
import jakarta.validation.Payload;

import gov.irs.boot.validation.constraints.validators.NotBlankForAwsValidator;

@Documented
@Constraint(validatedBy = NotBlankForAwsValidator.class)
@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
public @interface NotBlankForAws {
    String message() default "{gov.irs.constraint.NotBlankForAws.message}";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
