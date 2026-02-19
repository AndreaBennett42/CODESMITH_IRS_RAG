package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.publishsc;

import java.util.List;
import java.util.UUID;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.IntNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.google.common.base.Charsets;
import lombok.SneakyThrows;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.stubbing.Answer;

import gov.irs.directfile.api.taxreturn.TaxReturnSubmissionRepository;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

import static gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.publishsc.PublishSubmissionConfirmationsEventHandler.SUBMISSION_IDS_KEY;
import static gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.publishsc.PublishSubmissionConfirmationsEventHandler.TAX_RETURN_SUBMISSION_RECEIPT_INFORMATIONS_KEY;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class PublishSubmissionConfirmationsEventHandlerTest {
    private PublishSubmissionConfirmationsEventHandler publishSubmissionConfirmationsEventHandler;

    @Mock
    private TaxReturnSubmissionRepository taxReturnSubmissionRepository;

    @Mock
    private SubmissionConfirmationMessageService submissionConfirmationMessageService;

    private final ObjectMapper objectMapper = new ObjectMapper();

    String publishSubmissionConfirmationsJson = "{\n" + "  \"key\": \"publish_submission_confirmations\",\n"
            + "  \"payload\": {\n"
            + "    \"taxReturnSubmissionReceiptInformations\": [\n"
            + "      {\n"
            + "        \"submissionId\": \"submissionId1\",\n"
            + "        \"receiptId\": \"receiptId1\"\n"
            + "      },\n"
            + "      {\n"
            + "        \"submissionId\": \"submissionId2\",\n"
            + "        \"receiptId\": \"receiptId2\"\n"
            + "      }\n"
            + "    ]\n"
            + "  }\n"
            + "}";

    @BeforeEach
    public void setup() {
        publishSubmissionConfirmationsEventHandler = new PublishSubmissionConfirmationsEventHandler(
                taxReturnSubmissionRepository, submissionConfirmationMessageService, objectMapper);
    }

    @Test
    @SneakyThrows
    void givenValidPayloadWithLessThan100SubmissionIds_whenHandleEvent_thenPublishesSingleMessage() {
        // Given
        JsonNode payload = objectMapper
                .readTree(publishSubmissionConfirmationsJson.getBytes(Charsets.UTF_8))
                .get("payload");

        TaxReturnSubmissionIdAndReceiptId taxReturnSubmissionIdAndReceiptId1 = new TaxReturnSubmissionIdAndReceiptId();
        taxReturnSubmissionIdAndReceiptId1.setSubmissionId("submissionId1");
        taxReturnSubmissionIdAndReceiptId1.setReceiptId("receiptId1");

        TaxReturnSubmissionIdAndReceiptId taxReturnSubmissionIdAndReceiptId2 = new TaxReturnSubmissionIdAndReceiptId();
        taxReturnSubmissionIdAndReceiptId2.setSubmissionId("submissionId2");
        taxReturnSubmissionIdAndReceiptId2.setReceiptId("receiptId2");

        TaxReturnSubmission taxReturnSubmission1 = new TaxReturnSubmission();
        taxReturnSubmission1.setTaxReturnId(UUID.fromString("00000000-0000-0000-0000-000000000111"));
        taxReturnSubmission1.setSubmissionId("submissionId1");

        TaxReturnSubmission taxReturnSubmission2 = new TaxReturnSubmission();
        taxReturnSubmission2.setTaxReturnId(UUID.fromString("00000000-0000-0000-0000-000000000222"));
        taxReturnSubmission2.setSubmissionId("submissionId2");

        List<TaxReturnSubmission> taxReturnSubmissions = List.of(taxReturnSubmission1, taxReturnSubmission2);
        when(taxReturnSubmissionRepository.findAllBySubmissionIds(List.of(
                        taxReturnSubmissionIdAndReceiptId1.getSubmissionId(),
                        taxReturnSubmissionIdAndReceiptId2.getSubmissionId())))
                .thenReturn(taxReturnSubmissions);

        // When
        publishSubmissionConfirmationsEventHandler.handleNotificationEvent(payload);

        // Then
        ArgumentCaptor<List<SubmissionConfirmationPayloadV2Entry>> argumentCaptor = ArgumentCaptor.forClass(List.class);
        verify(submissionConfirmationMessageService, times(1))
                .publishSubmissionConfirmationPayloadV2(argumentCaptor.capture());

        List<SubmissionConfirmationPayloadV2Entry> submissionConfirmationPayloadV2Entries = argumentCaptor.getValue();

        assertEquals(2, submissionConfirmationPayloadV2Entries.size());

        assertEquals(
                taxReturnSubmissionIdAndReceiptId1.getSubmissionId(),
                submissionConfirmationPayloadV2Entries
                        .get(0)
                        .getTaxReturnSubmissionReceipt()
                        .getSubmissionId());
        assertEquals(
                taxReturnSubmission1.getTaxReturnId(),
                submissionConfirmationPayloadV2Entries
                        .get(0)
                        .getTaxReturnSubmissionReceipt()
                        .getTaxReturnId());
        assertNotNull(submissionConfirmationPayloadV2Entries
                .get(0)
                .getTaxReturnSubmissionReceipt()
                .getSubmissionReceivedAt());

        assertEquals(
                taxReturnSubmissionIdAndReceiptId2.getSubmissionId(),
                submissionConfirmationPayloadV2Entries
                        .get(1)
                        .getTaxReturnSubmissionReceipt()
                        .getSubmissionId());
        assertEquals(
                taxReturnSubmission2.getTaxReturnId(),
                submissionConfirmationPayloadV2Entries
                        .get(1)
                        .getTaxReturnSubmissionReceipt()
                        .getTaxReturnId());
        assertNotNull(submissionConfirmationPayloadV2Entries
                .get(1)
                .getTaxReturnSubmissionReceipt()
                .getSubmissionReceivedAt());

        submissionConfirmationPayloadV2Entries.forEach(entry -> {
            assertEquals(SubmissionEventTypeEnum.SUBMITTED, entry.getEventType());
        });
    }

    @Test
    @SneakyThrows
    void givenValidPayloadWithGreater250SubmissionIds_whenHandleEvent_thenPublishes3Messages() {
        // Given
        ObjectNode payload = (ObjectNode) objectMapper
                .readTree(publishSubmissionConfirmationsJson.getBytes(Charsets.UTF_8))
                .get("payload");

        ArrayNode submisssionIdsArrayNode = (ArrayNode) payload.get(TAX_RETURN_SUBMISSION_RECEIPT_INFORMATIONS_KEY);
        for (int i = 0; i < 250; i++) {
            TaxReturnSubmissionIdAndReceiptId t = new TaxReturnSubmissionIdAndReceiptId();
            t.setSubmissionId("submissionId" + i);
            t.setReceiptId("receiptId" + i);
            submisssionIdsArrayNode.add(objectMapper.valueToTree(t));
        }

        payload.set(TAX_RETURN_SUBMISSION_RECEIPT_INFORMATIONS_KEY, submisssionIdsArrayNode);
        assertEquals(
                252,
                payload.get(TAX_RETURN_SUBMISSION_RECEIPT_INFORMATIONS_KEY)
                        .size()); // 252 because there were already 2 in the default json payload

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

        // When
        publishSubmissionConfirmationsEventHandler.handleNotificationEvent(payload);

        // Then
        ArgumentCaptor<List<SubmissionConfirmationPayloadV2Entry>> argumentCaptor = ArgumentCaptor.forClass(List.class);
        verify(taxReturnSubmissionRepository, times(3)).findAllBySubmissionIds(anyList());
        verify(submissionConfirmationMessageService, times(3))
                .publishSubmissionConfirmationPayloadV2(argumentCaptor.capture());

        assertEquals(100, argumentCaptor.getAllValues().get(0).size());
        assertEquals(100, argumentCaptor.getAllValues().get(1).size());
        assertEquals(52, argumentCaptor.getAllValues().get(2).size());
    }

    @Test
    @SneakyThrows
    void givenValidPayloadWithGreater300SubmissionIdsAndCustomBatchSizeOf150_whenHandleEvent_thenPublishes2Messages() {
        // Given

        ObjectNode payload = (ObjectNode) objectMapper
                .readTree(publishSubmissionConfirmationsJson.getBytes(Charsets.UTF_8))
                .get("payload");

        ArrayNode submisssionIdsArrayNode = (ArrayNode) payload.get(TAX_RETURN_SUBMISSION_RECEIPT_INFORMATIONS_KEY);
        for (int i = 0; i < 298; i++) {
            TaxReturnSubmissionIdAndReceiptId t = new TaxReturnSubmissionIdAndReceiptId();
            t.setSubmissionId("submissionId" + i);
            t.setReceiptId("receiptId" + i);
            submisssionIdsArrayNode.add(objectMapper.valueToTree(t));
        }

        payload.set(SUBMISSION_IDS_KEY, submisssionIdsArrayNode);
        assertEquals(
                300,
                payload.get(SUBMISSION_IDS_KEY).size()); // 252 because there were already 2 in the default json payload
        payload.set("batchSize", new IntNode(150));

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

        // When
        publishSubmissionConfirmationsEventHandler.handleNotificationEvent(payload);

        // Then
        ArgumentCaptor<List<SubmissionConfirmationPayloadV2Entry>> argumentCaptor = ArgumentCaptor.forClass(List.class);
        verify(taxReturnSubmissionRepository, times(2)).findAllBySubmissionIds(anyList());
        verify(submissionConfirmationMessageService, times(2))
                .publishSubmissionConfirmationPayloadV2(argumentCaptor.capture());

        assertEquals(150, argumentCaptor.getAllValues().get(0).size());
        assertEquals(150, argumentCaptor.getAllValues().get(1).size());
    }

    @Test
    @SneakyThrows
    void givenInvalidPayload_whenHandleEvent_thenDoesNotPublishMessage() {
        // Given
        ObjectNode payload = (ObjectNode) objectMapper
                .readTree(publishSubmissionConfirmationsJson.getBytes(Charsets.UTF_8))
                .get("payload");

        payload.remove(TAX_RETURN_SUBMISSION_RECEIPT_INFORMATIONS_KEY);

        // When
        publishSubmissionConfirmationsEventHandler.handleNotificationEvent(payload);

        // Then
        verify(taxReturnSubmissionRepository, never()).findAllBySubmissionIds(anyList());
        verify(submissionConfirmationMessageService, never()).publishSubmissionConfirmationPayloadV2(anyList());
    }
}
