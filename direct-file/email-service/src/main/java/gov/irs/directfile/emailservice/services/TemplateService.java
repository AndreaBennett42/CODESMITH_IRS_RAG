package gov.irs.directfile.emailservice.services;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.EnumMap;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

import gov.irs.directfile.emailservice.domain.Template;
import gov.irs.directfile.emailservice.exceptions.MissingTemplateException;
import gov.irs.directfile.models.LepLanguage;
import gov.irs.directfile.models.email.HtmlTemplate;

@Service
@Slf4j
public class TemplateService {
    private static final String FILE_EXT_HTML = "html";
    private static final String FILE_EXT_JSON = "json";
    private static final String TEMPLATE_DOES_NOT_EXIST = "Does not exist: %s";
    private static final String TEMPLATE_FAILED_TO_READ = "Failed to read: %s";
    private static final String TEMPLATE_FILE_PATH = "classpath:templates/%s/%s.%s";
    private static final String TEMPLATE_LOADED = "Loaded: %s";

    Map<LepLanguage, Map<HtmlTemplate, Template>> templates;

    @SneakyThrows
    public TemplateService(ResourceLoader resourceLoader) {
        templates = new EnumMap<>(LepLanguage.class);
        loadTemplates(resourceLoader);
    }

    public void loadTemplates(ResourceLoader resourceLoader) throws IOException, URISyntaxException {
        log.info("Loading up HTML email templates");

        for (LepLanguage language : LepLanguage.values()) {
            if (language.isEnabled()) {
                final Map<HtmlTemplate, Template> map = new EnumMap<>(HtmlTemplate.class);
                final String code = language.toCode();

                for (HtmlTemplate htmlTemplate : HtmlTemplate.values()) {
                    final String fileName = htmlTemplate.toString().toLowerCase();

                    // TODO deprecate json file; move subject into message.properties
                    final Resource json = resourceLoader.getResource(
                            String.format(TEMPLATE_FILE_PATH, code, fileName, FILE_EXT_JSON));
                    final Resource html = resourceLoader.getResource(
                            String.format(TEMPLATE_FILE_PATH, code, fileName, FILE_EXT_HTML));

                    if (json.isReadable() && html.isReadable()) {
                        final Template template = new ObjectMapper()
                                .readValue(json.getContentAsString(StandardCharsets.UTF_8), Template.class);

                        // TODO refactor to read subject from MessageSource
                        // template.subject =
                        // messageSource.getMessage(htmlTemplate.getSubjectMessageKey(),
                        // locale);
                        template.setBody(html.getContentAsString(StandardCharsets.UTF_8));
                        map.put(htmlTemplate, template);
                        log.info(String.format(
                                TEMPLATE_LOADED, String.format(TEMPLATE_FILE_PATH, code, fileName, FILE_EXT_JSON)));
                        log.info(String.format(
                                TEMPLATE_LOADED, String.format(TEMPLATE_FILE_PATH, code, fileName, FILE_EXT_HTML)));
                    } else {
                        logErrors(code, fileName, FILE_EXT_JSON, json);
                        logErrors(code, fileName, FILE_EXT_HTML, json);
                    }
                }

                if (!map.isEmpty()) {
                    templates.put(language, map);
                }
            }
        }
    }

    public void logErrors(String code, String fileName, String fileExt, Resource resource) {
        if (!resource.exists()) {
            log.error(
                    String.format(TEMPLATE_DOES_NOT_EXIST, String.format(TEMPLATE_FILE_PATH, code, fileName, fileExt)));
        } else if (!resource.isReadable()) {
            log.error(
                    String.format(TEMPLATE_FAILED_TO_READ, String.format(TEMPLATE_FILE_PATH, code, fileName, fileExt)));
        }
    }

    public Template getTemplate(String languageCode, HtmlTemplate template) throws MissingTemplateException {
        return getTemplate(LepLanguage.fromCode(languageCode), template);
    }

    public Template getTemplate(LepLanguage language, HtmlTemplate template) throws MissingTemplateException {
        String code = language.toCode();
        log.info(String.format("getting template %s/%s", code, template));

        if (templates.containsKey(language)) {
            Map<HtmlTemplate, Template> templateCollection = templates.get(language);
            if (templateCollection.containsKey(template)) {
                log.info(String.format("found template %s/%s", code, template));
                return templateCollection.get(template);
            } else {
                throw new MissingTemplateException(
                        String.format("Could not find template %s for language code %s", template, code));
            }
        } else {
            throw new MissingTemplateException(String.format("Could not find language code: %s", code));
        }
    }
}
