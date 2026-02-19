package gov.irs.directfile.submit.service.handlers.dispatch;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.models.message.dispatch.VersionedDispatchMessage;
import gov.irs.directfile.models.message.dispatch.payload.AbstractDispatchPayload;

@Service
@Slf4j
public class UnsupportedMessageVersionHandler implements DispatchHandler {
    @Override
    public void handleDispatchMessage(VersionedDispatchMessage<AbstractDispatchPayload> payload) {
        log.error("Unable to process Dispatch Message. Headers: {} ", payload.getHeaders());
    }
}
