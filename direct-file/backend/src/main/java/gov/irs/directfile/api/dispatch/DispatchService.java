package gov.irs.directfile.api.dispatch;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Locale;
import java.util.UUID;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import jakarta.transaction.Transactional;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.config.MessageQueueConfigurationProperties;
import gov.irs.directfile.api.loaders.service.FactGraphService;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.models.Dispatch;

@Service
@Transactional
@EnableConfigurationProperties(MessageQueueConfigurationProperties.class)
public class DispatchService {
    private final FactGraphService factGraphService;
    private final DispatchQueueService dispatchQueueService;
    private final boolean isSqsMessageSendingEnabled;

    private final ObjectMapper mapper = new ObjectMapper();

    public DispatchService(
            FactGraphService factGraphService,
            DispatchQueueService dispatchQueueService,
            MessageQueueConfigurationProperties messageQueueConfigurationProperties) {
        this.factGraphService = factGraphService;
        this.dispatchQueueService = dispatchQueueService;
        this.isSqsMessageSendingEnabled = messageQueueConfigurationProperties.isSqsMessageSendingEnabled();
        JavaTimeModule module = new JavaTimeModule();
        DateFormat df = new SimpleDateFormat("yyyy-dd-MM HH:mm:ss", Locale.US);
        this.mapper.setDateFormat(df);
        this.mapper.registerModule(module); // Java 8 time not registered by default
    }

    public void enqueue(UUID userId, TaxReturn taxReturn, DispatchContext context) {
        Dispatch dispatch = new Dispatch(
                userId,
                taxReturn.getId(),
                context.pathToManifestXml,
                context.pathToUserContext,
                context.pathToSubmissionXml,
                context.submissionId);
        if (isSqsMessageSendingEnabled) {
            dispatchQueueService.enqueue(dispatch);
        }
    }

    public void internalOnlyEnqueue(UUID userId, TaxReturn taxReturn, DispatchContext context) {
        Dispatch dispatch = new Dispatch(
                userId,
                taxReturn.getId(),
                context.pathToManifestXml,
                context.pathToUserContext,
                context.pathToSubmissionXml,
                context.submissionId);
        if (isSqsMessageSendingEnabled) {
            dispatchQueueService.enqueue(dispatch);
        }
    }
}
