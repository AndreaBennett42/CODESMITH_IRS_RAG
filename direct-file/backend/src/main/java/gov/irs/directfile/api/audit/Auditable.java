package gov.irs.directfile.api.audit;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import gov.irs.directfile.api.events.EventId;
import gov.irs.directfile.api.events.UserType;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Auditable {

    EventId event();

    UserType type() default UserType.SYS;
}
