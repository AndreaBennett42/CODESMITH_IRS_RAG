package gov.irs.directfile.emailservice.services;

import java.util.EnumMap;
import java.util.Map;

import lombok.SneakyThrows;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.ResourceLoader;

import gov.irs.directfile.emailservice.domain.Template;
import gov.irs.directfile.models.email.HtmlTemplate;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;

class TemplateServiceTest {
    ResourceLoader resourceLoader = new DefaultResourceLoader();

    @SneakyThrows
    @Test
    void CanGetAllExpectedTemplates() {
        TemplateService service = new TemplateService(resourceLoader);

        Map<HtmlTemplate, String> subjects = new EnumMap<>(HtmlTemplate.class);
        subjects.put(HtmlTemplate.ACCEPTED, "Accepted: Your 2024 tax return");
        subjects.put(HtmlTemplate.REJECTED, "Rejected: Your 2024 tax return");
        subjects.put(HtmlTemplate.SUBMITTED, "Submitted: Your 2024 tax return");
        subjects.put(HtmlTemplate.PRE_SUBMISSION_ERROR, "Unable to submit your 2024 tax return");
        subjects.put(HtmlTemplate.POST_SUBMISSION_ERROR, "Problem submitting your 2024 federal tax return");
        subjects.put(HtmlTemplate.ERROR_RESOLVED, "Action needed: Submit your 2024 federal tax return");
        subjects.put(HtmlTemplate.REMINDER_SUBMIT, "Reminder: Submit your 2024 federal tax return by April 15");
        subjects.put(HtmlTemplate.REMINDER_RESUBMIT, "Reminder: Resubmit your rejected federal tax return");
        subjects.put(HtmlTemplate.REMINDER_STATE, "Reminder: File your state taxes");
        subjects.put(
                HtmlTemplate.NON_COMPLETION_SURVEY,
                "Give feedback on IRS Direct File/Comparta su opinión sobre Direct File del IRS");
        for (HtmlTemplate htmlTemplate : HtmlTemplate.values()) {
            Template template = service.getTemplate("en", htmlTemplate);
            assertThat(template.getSubject()).isEqualTo(subjects.get(htmlTemplate));
            assertThat(template.getBody()).startsWith("<!DOCTYPE html>");
        }
    }

    @Test
    void WillFailWhenAttemptingToGetAnUnknownLanguage() {
        TemplateService s = new TemplateService(resourceLoader);
        assertThrows(IllegalArgumentException.class, () -> {
            s.getTemplate("rs", HtmlTemplate.ACCEPTED);
        });
    }
}
