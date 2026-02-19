package gov.irs.directfile.api.taxreturn.submissions;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class SendEmailQueueStatusEnumTest {
    @Test
    public void givenAValidStatus_whenFindingCaseAgnosticValueFromString_thenReturnsEquivalentEnumStatus() {
        SendEmailQueueStatusEnum status = SendEmailQueueStatusEnum.valueOfIgnoreCase("AcCePtEd");
        assertEquals(SendEmailQueueStatusEnum.ACCEPTED, status);
    }

    @Test
    public void givenAnInvalidStatus_whenFindingCaseAgnosticValueFromString_thenThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            SendEmailQueueStatusEnum.valueOfIgnoreCase("invalid");
        });
    }
}
