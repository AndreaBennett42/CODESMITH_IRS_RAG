package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification;

import java.io.IOException;
import java.util.List;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.google.common.base.Charsets;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.stubbing.Answer;

import gov.irs.directfile.api.taxreturn.TaxReturnService;
import gov.irs.directfile.api.taxreturn.TaxReturnSubmissionRepository;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class PublishDispatchMessageEventHandlerTest {
    ObjectMapper mapper = new ObjectMapper();
    PublishDispatchMessageEventHandler publishDispatchMessageEventHandler;

    @Mock
    private TaxReturnService taxReturnService;

    @Mock
    private TaxReturnSubmissionRepository taxReturnSubmissionRepository;

    String publishDispatchMessageJson =
            "{\"key\":\"publish_dispatch_queue_messages\",\"payload\":{\"submissionIds\":[\"1111\",\"2222\"]}}";

    @BeforeEach
    void setup() {
        publishDispatchMessageEventHandler =
                new PublishDispatchMessageEventHandler(taxReturnService, taxReturnSubmissionRepository);
    }

    @Test
    void
            givenValidJson_whenHandleMessage_parsesSubmissionIdsAndFetchesTaxReturnSubmissionAndPublishesDispatchesMessageForEachOne()
                    throws IOException {
        JsonNode payload = mapper.readTree(publishDispatchMessageJson.getBytes(Charsets.UTF_8))
                .get("payload");

        String submissionId1 = "1111";
        String submissionId2 = "2222";

        TaxReturnSubmission taxReturnSubmission1 = new TaxReturnSubmission();
        taxReturnSubmission1.setSubmissionId(submissionId1);
        TaxReturn taxReturn1 = mock(TaxReturn.class);
        taxReturnSubmission1.setTaxReturn(taxReturn1);

        TaxReturnSubmission taxReturnSubmission2 = new TaxReturnSubmission();
        taxReturnSubmission2.setSubmissionId(submissionId2);
        TaxReturn taxReturn2 = mock(TaxReturn.class);
        taxReturnSubmission2.setTaxReturn(taxReturn2);

        List<TaxReturnSubmission> taxReturnSubmissions = List.of(taxReturnSubmission1, taxReturnSubmission2);
        when(taxReturnSubmissionRepository.findAllBySubmissionIds(eq(List.of(submissionId1, submissionId2))))
                .thenReturn(taxReturnSubmissions);

        publishDispatchMessageEventHandler.handleNotificationEvent(payload);

        verify(taxReturnService, times(0)).stubEnqueueDispatch();
    }

    @Test
    void
            givenValidJsonWith152SubmissionIds_whenHandleMessage_parsesSubmissionIdsIntoBatchesOf50AndFetchesTaxReturnSubmissionAndPublishesDispatchesMessageForEachOne()
                    throws IOException {
        ObjectNode payload = (ObjectNode) mapper.readTree(publishDispatchMessageJson.getBytes(Charsets.UTF_8))
                .get("payload");

        ArrayNode submisssionIdsArrayNode = (ArrayNode) payload.get("submissionIds");
        for (int i = 0; i < 150; i++) {
            submisssionIdsArrayNode.add("submissionId" + i);
        }

        payload.set("submissionIds", submisssionIdsArrayNode);
        assertEquals(152, payload.get("submissionIds").size());

        // Dynamically mock the return value of findAllBySubmissionIds to return a list of size equal to the size of the
        // list of submissionIds passed to it.
        when(taxReturnSubmissionRepository.findAllBySubmissionIds(anyList()))
                .thenAnswer((Answer<List<TaxReturnSubmission>>) invocation -> {
                    List<String> submissionIds = invocation.getArgument(0);

                    return submissionIds.stream()
                            .map(s -> {
                                TaxReturnSubmission taxReturnSubmission = new TaxReturnSubmission();
                                taxReturnSubmission.setSubmissionId(s);
                                return taxReturnSubmission;
                            })
                            .toList();
                });

        publishDispatchMessageEventHandler.handleNotificationEvent(payload);

        verify(taxReturnSubmissionRepository, times(4)).findAllBySubmissionIds(anyList());
    }

    @Test
    void givenInValidJson_whenHandleMessage_doesNothing() throws IOException {
        ObjectNode payload = (ObjectNode) mapper.readTree(publishDispatchMessageJson.getBytes(Charsets.UTF_8))
                .get("payload");

        payload.set("submissionIds", mapper.createArrayNode());
        publishDispatchMessageEventHandler.handleNotificationEvent(payload);

        verify(taxReturnService, times(0)).stubEnqueueDispatch();
    }

    @Test
    void givenJsonWithEmptySubmissionIdsArray_whenHandleMessage_doesNothing() {
        publishDispatchMessageEventHandler.handleNotificationEvent(mapper.createObjectNode());

        verify(taxReturnService, times(0)).stubEnqueueDispatch();
    }

    @Test
    void givenNoTaxReturnSubmissionsFound_whenHandleMessage_doesNotDispatchMessages() throws IOException {
        JsonNode payload = mapper.readTree(publishDispatchMessageJson.getBytes(Charsets.UTF_8))
                .get("payload");

        when(taxReturnSubmissionRepository.findAllBySubmissionIds(anyList())).thenReturn(List.of());

        publishDispatchMessageEventHandler.handleNotificationEvent(payload);

        verify(taxReturnService, times(0)).stubEnqueueDispatch();
    }
}
