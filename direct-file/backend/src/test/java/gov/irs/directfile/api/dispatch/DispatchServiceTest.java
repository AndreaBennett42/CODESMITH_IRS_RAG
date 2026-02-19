package gov.irs.directfile.api.dispatch;

import java.util.Map;
import java.util.UUID;

import com.fasterxml.jackson.core.JsonProcessingException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.api.config.MessageQueueConfigurationProperties;
import gov.irs.directfile.api.errors.InvalidDataException;
import gov.irs.directfile.api.loaders.service.FactGraphService;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.models.Dispatch;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DispatchServiceTest {
    DispatchService dispatchService;

    @Mock
    private FactGraphService factGraphService;

    @Mock
    MessageQueueConfigurationProperties messageQueueConfigurationProperties;

    @Mock
    private DispatchQueueService dispatchQueueService;

    private void setup(boolean isSqsMessageSendingEnabled) {
        when(messageQueueConfigurationProperties.isSqsMessageSendingEnabled()).thenReturn(isSqsMessageSendingEnabled);
        dispatchService =
                new DispatchService(factGraphService, dispatchQueueService, messageQueueConfigurationProperties);
    }

    @Test
    void enqueue_ifSqsMessageSendingIsNotEnabled_savesDispatch() throws InvalidDataException, JsonProcessingException {
        setup(false);

        TaxReturn taxReturn = new TaxReturn();
        taxReturn.setFacts(Map.of());

        DispatchContext context = new DispatchContext("", "", "", "");
        dispatchService.enqueue(UUID.randomUUID(), taxReturn, context);

        verify(dispatchQueueService, never()).enqueue(any());
    }

    @Test
    void enqueue_SendsDispatchMessageToMessageQueue() throws InvalidDataException, JsonProcessingException {
        setup(true);

        TaxReturn taxReturn = new TaxReturn();
        taxReturn.setFacts(Map.of());

        DispatchContext context = new DispatchContext("", "", "", "");
        dispatchService.enqueue(UUID.randomUUID(), taxReturn, context);

        verify(dispatchQueueService).enqueue(any(Dispatch.class));
    }
}
