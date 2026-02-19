package gov.irs.boot.validation.constraints;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import jakarta.validation.Constraint;
import jakarta.validation.Payload;

import gov.irs.boot.validation.constraints.validators.NotBlankForNotAwsValidator;

@Documented
@Constraint(validatedBy = NotBlankForNotAwsValidator.class)
@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
public @interface NotBlankForNotAws {
    String message() default "{gov.irs.constraint.NotBlankForNotAws.message}";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
