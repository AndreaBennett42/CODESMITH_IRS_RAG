package gov.irs.directfile.submit.domain;

import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import javax.xml.datatype.DatatypeFactory;
import javax.xml.datatype.XMLGregorianCalendar;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.mef.services.transmitter.mtom.SendSubmissionsResult;

import gov.irs.directfile.audit.events.TinType;
import gov.irs.directfile.models.TaxReturnIdAndSubmissionId;
import gov.irs.directfile.models.message.confirmation.payload.SubmissionConfirmationPayloadV2Entry;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class SubmittedDataContainerTest {
    SubmittedDataContainer submittedDataContainer;

    UserContextData userContextData1 = new UserContextData(
            "submissionId1",
            "userId1",
            "b30b08d1-d14d-49c5-b902-ddee26111111",
            "userTin1",
            TinType.INDIVIDUAL,
            "remoteAddress1",
            "signDate1");
    UserContextData userContextData2 = new UserContextData(
            "submissionId2",
            "userId2",
            "b30b08d1-d14d-49c5-b902-ddee26222222",
            "userTin",
            TinType.INDIVIDUAL,
            "remoteAddress2",
            "signDate2");

    @Mock
    SendSubmissionsResult submissionsResult;

    @Mock
    SendSubmissionsResultWrapper sendSubmissionsResultWrapper;

    @Mock
    SubmissionReceiptGrpWrapper submissionReceiptGrpWrapper1;

    @Mock
    SubmissionReceiptGrpWrapper submissionReceiptGrpWrapper2;

    XMLGregorianCalendar date1 = createXMLGregorianCalendar();
    XMLGregorianCalendar date2 = createXMLGregorianCalendar();

    @BeforeEach
    public void setup() {
        List<UserContextData> userContexts = List.of(userContextData1, userContextData2);
        SubmissionBatch submissionBatch = null;
        submittedDataContainer =
                new SubmittedDataContainer(userContexts, sendSubmissionsResultWrapper, submissionBatch);
    }

    @Test
    public void
            getTaxReturnIdAndSubmissionIds_givenUserContextData_mapsUserContextDataToListOfTaxReturnIdAndSubmissionId() {
        List<TaxReturnIdAndSubmissionId> result = submittedDataContainer.getTaxReturnIdAndSubmissionIds();

        assertEquals(2, result.size());
        assertEquals(userContextData1.getSubmissionId(), result.get(0).getSubmissionId());
        assertEquals(
                userContextData1.getTaxReturnId(), String.valueOf(result.get(0).getTaxReturnId()));

        assertEquals(userContextData2.getSubmissionId(), result.get(1).getSubmissionId());
        assertEquals(
                userContextData2.getTaxReturnId(), String.valueOf(result.get(1).getTaxReturnId()));
    }

    @Test
    public void getTaxReturnIdAndSubmissionIds_givenEmptyUserContextData_returnsEmptyList() {
        submittedDataContainer.userContexts = List.of();
        List<TaxReturnIdAndSubmissionId> result = submittedDataContainer.getTaxReturnIdAndSubmissionIds();

        assertEquals(0, result.size());
    }

    @Test
    public void
            getTaxReturnSubmissionReceipts_givenSendSubmissionsResultWrapper_mapsUserContextDataToListOfTaxReturnSubmissionReceipt() {
        when(submissionReceiptGrpWrapper1.getReceiptId()).thenReturn("receiptId1");
        when(submissionReceiptGrpWrapper1.getSubmissionId()).thenReturn("submissionId1");
        when(submissionReceiptGrpWrapper1.getSubmissionReceivedTs()).thenReturn(date1);

        when(submissionReceiptGrpWrapper2.getReceiptId()).thenReturn("receiptId2");
        when(submissionReceiptGrpWrapper2.getSubmissionId()).thenReturn("submissionId2");
        when(submissionReceiptGrpWrapper2.getSubmissionReceivedTs()).thenReturn(date2);

        when(sendSubmissionsResultWrapper.getReceipts())
                .thenReturn(List.of(submissionReceiptGrpWrapper1, submissionReceiptGrpWrapper2));

        List<SubmissionConfirmationPayloadV2Entry> result =
                submittedDataContainer.getSuccessSubmissionConfirmationPayloadV2Entries();

        assertEquals(2, result.size());

        SubmissionConfirmationPayloadV2Entry entry1 = result.getFirst();
        assertEquals("submissionId1", entry1.getTaxReturnSubmissionReceipt().getSubmissionId());
        assertEquals("receiptId1", entry1.getTaxReturnSubmissionReceipt().getReceiptId());
        assertEquals(
                userContextData1.getTaxReturnId(),
                String.valueOf(entry1.getTaxReturnSubmissionReceipt().getTaxReturnId()));
        assertEquals(
                date1.toGregorianCalendar().getTime(),
                entry1.getTaxReturnSubmissionReceipt().getSubmissionReceivedAt());
        assertEquals(SubmissionEventTypeEnum.SUBMITTED, entry1.getEventType());
        assertTrue(entry1.getMetadata().isEmpty());

        SubmissionConfirmationPayloadV2Entry entry2 = result.get(1);
        assertEquals("submissionId2", entry2.getTaxReturnSubmissionReceipt().getSubmissionId());
        assertEquals("receiptId2", entry2.getTaxReturnSubmissionReceipt().getReceiptId());
        assertEquals(
                userContextData2.getTaxReturnId(),
                String.valueOf(entry2.getTaxReturnSubmissionReceipt().getTaxReturnId()));
        assertEquals(
                date2.toGregorianCalendar().getTime(),
                entry2.getTaxReturnSubmissionReceipt().getSubmissionReceivedAt());
        assertEquals(SubmissionEventTypeEnum.SUBMITTED, entry2.getEventType());
        assertTrue(entry2.getMetadata().isEmpty());
    }

    @Test
    public void getTaxReturnSubmissionReceipts_givenEmptySendSubmissionsResultWrapper_returnsEmptyList() {
        when(sendSubmissionsResultWrapper.getReceipts()).thenReturn(List.of());
        List<SubmissionConfirmationPayloadV2Entry> result =
                submittedDataContainer.getSuccessSubmissionConfirmationPayloadV2Entries();

        assertEquals(0, result.size());
    }

    private XMLGregorianCalendar createXMLGregorianCalendar() {
        OffsetDateTime date = OffsetDateTime.now();
        // NOTE: ISO_OFFSET_DATE_TIME includes fractional seconds.
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("uuuu-MM-dd'T'HH:mm:ssXXX");
        var lex = date.format(formatter);
        return DatatypeFactory.newDefaultInstance().newXMLGregorianCalendar(lex);
    }
}
