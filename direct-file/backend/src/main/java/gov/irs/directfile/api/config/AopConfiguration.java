package gov.irs.directfile.api.config;

import java.lang.reflect.Method;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.AfterReturning;
import org.aspectj.lang.annotation.AfterThrowing;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.context.annotation.EnableAspectJAutoProxy;
import org.springframework.stereotype.Component;

import gov.irs.directfile.api.audit.AuditEventContextHolder;
import gov.irs.directfile.api.audit.AuditLogElement;
import gov.irs.directfile.api.audit.AuditService;
import gov.irs.directfile.api.audit.Auditable;
import gov.irs.directfile.api.events.Event;
import gov.irs.directfile.api.events.EventPrincipal;
import gov.irs.directfile.api.events.EventStatus;
import gov.irs.directfile.api.events.SystemEventPrincipal;
import gov.irs.directfile.api.events.UserType;

@EnableAspectJAutoProxy
@Aspect
@Component
@AllArgsConstructor
@Slf4j
@SuppressWarnings({"PMD.UnusedFormalParameter"})
public class AopConfiguration {
    private final AuditService auditService;
    private final AuditEventContextHolder auditEventContextHolder;

    @Pointcut("@annotation(gov.irs.directfile.api.audit.Auditable)")
    public void auditableMethods() {}

    @AfterReturning("auditableMethods()")
    public void logAfterAuditableMethod(JoinPoint jp) {
        MethodSignature signature = (MethodSignature) jp.getSignature();

        // annotations
        Method method = signature.getMethod();
        Auditable auditableAnnotation = method.getAnnotation(Auditable.class);

        auditService.addAuditPropertiesToMDC(Event.builder()
                .eventId(auditableAnnotation.event())
                .eventStatus(EventStatus.SUCCESS)
                .eventPrincipal(createEventPrincipal(auditableAnnotation.type()))
                .build());
    }

    @AfterThrowing(value = "auditableMethods()", throwing = "ex")
    public void logAfterAuditableMethodException(JoinPoint jp, Throwable ex) {
        MethodSignature signature = (MethodSignature) jp.getSignature();

        // annotations
        Method method = signature.getMethod();
        Auditable auditableAnnotation = method.getAnnotation(Auditable.class);

        auditEventContextHolder.addValueToEventDetailMap(AuditLogElement.DetailElement.MESSAGE, ex.getMessage());
        auditService.addAuditPropertiesToMDC(Event.builder()
                .eventId(auditableAnnotation.event())
                .eventStatus(EventStatus.FAILURE)
                .eventPrincipal(createEventPrincipal(auditableAnnotation.type()))
                .eventErrorMessage(ex.getClass().getName())
                .build());
    }

    private EventPrincipal createEventPrincipal(UserType userType) {
        return new SystemEventPrincipal();
    }
}
