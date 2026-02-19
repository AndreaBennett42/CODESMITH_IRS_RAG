package gov.irs.directfile.submit.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.jms.Message;
import jakarta.jms.MessageListener;
import jakarta.jms.TextMessage;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import gov.irs.directfile.audit.AuditLogElement;
import gov.irs.directfile.models.message.dispatch.VersionedDispatchMessage;
import gov.irs.directfile.models.message.dispatch.payload.AbstractDispatchPayload;

@Service
@Slf4j
public class UserSubmissionConsumer implements MessageListener {
    private final DispatchMessageRouter dispatchMessageRouter;
    private final ObjectMapper objectMapper;

    @Autowired
    public UserSubmissionConsumer(DispatchMessageRouter dispatchMessageRouter, ObjectMapper objectMapper) {
        this.dispatchMessageRouter = dispatchMessageRouter;
        this.objectMapper = objectMapper;
    }

    @Override
    public void onMessage(Message message) {
        String taxReturnId = "";
        try {
            taxReturnId = message.getStringProperty("TAX-RETURN-ID");
            MDC.put(AuditLogElement.taxReturnId.toString(), taxReturnId);
            log.info("onMessage called, tax-return-id: {}", taxReturnId);

            String rawText = ((TextMessage) message).getText();
            VersionedDispatchMessage<AbstractDispatchPayload> versionedDispatchMessage =
                    objectMapper.readValue(rawText, new TypeReference<>() {});
            dispatchMessageRouter.handleDispatchMessage(versionedDispatchMessage);

            message.acknowledge();
        } catch (Exception ex) {
            MDC.put(AuditLogElement.taxReturnId.toString(), taxReturnId);
            log.error(
                    "unable to process a message from the queue, tax-return-id: {}, {}",
                    taxReturnId,
                    ex.getClass().getName(),
                    ex);
        } finally {
            MDC.clear();
        }
    }
}
