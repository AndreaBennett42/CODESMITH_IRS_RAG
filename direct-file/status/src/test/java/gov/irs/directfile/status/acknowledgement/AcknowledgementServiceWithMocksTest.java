package gov.irs.directfile.status.acknowledgement;

import java.util.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.*;
import static org.mockito.Mockito.times;

// AcknowledgementServiceTest is a @SpringBootTest
// This test uses mocks for lighter weight tests that don't need to test integration details as heavily.
@ExtendWith(MockitoExtension.class)
public class AcknowledgementServiceWithMocksTest {

    @Mock
    private TaxReturnSubmissionRepository taxReturnSubmissionRepository;

    @Mock
    private CompletedAcknowledgementRepository completedRepo;

    @InjectMocks
    AcknowledgementService acknowledgementService;

    @Test
    void getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission_returnsNullIfNoSubmissionIdsExist() {
        // Given
        var taxReturnId = UUID.randomUUID();

        when(taxReturnSubmissionRepository.getLatestAcceptedSubmissionIdForTaxReturnId(taxReturnId))
                .thenReturn(Optional.empty());
        when(taxReturnSubmissionRepository.getLatestSubmissionIdByTaxReturnId(taxReturnId))
                .thenReturn(Optional.empty());

        // When
        String submissionId =
                acknowledgementService.getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission(taxReturnId);

        // Then
        verify(taxReturnSubmissionRepository, times(1)).getLatestAcceptedSubmissionIdForTaxReturnId(taxReturnId);
        verify(taxReturnSubmissionRepository, times(1)).getLatestSubmissionIdByTaxReturnId(taxReturnId);
        assertNull(submissionId);
    }

    @Test
    void
            getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission_justGetsTheLatestSubmissionIdIfNoAcceptedSubmissionIsFound() {
        // Given
        var taxReturnId = UUID.randomUUID();
        var latestSubmissionId = "theLatestSubmissionId";

        when(taxReturnSubmissionRepository.getLatestAcceptedSubmissionIdForTaxReturnId(taxReturnId))
                .thenReturn(Optional.empty());
        when(taxReturnSubmissionRepository.getLatestSubmissionIdByTaxReturnId(taxReturnId))
                .thenReturn(Optional.of(latestSubmissionId));

        // When
        String submissionId =
                acknowledgementService.getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission(taxReturnId);

        // Then
        verify(taxReturnSubmissionRepository, times(1)).getLatestAcceptedSubmissionIdForTaxReturnId(taxReturnId);
        verify(taxReturnSubmissionRepository, times(1)).getLatestSubmissionIdByTaxReturnId(taxReturnId);
        assertEquals(latestSubmissionId, submissionId);
    }

    @Test
    void
            getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission_getsTheLatestAcceptedSubmissionIdEvenIfItIsNotTheLatest() {
        // Given
        var taxReturnId = UUID.randomUUID();
        var latestAcceptedSubmissionId = "theLatestAcceptedSubmissionId";

        when(taxReturnSubmissionRepository.getLatestAcceptedSubmissionIdForTaxReturnId(taxReturnId))
                .thenReturn(Optional.of(latestAcceptedSubmissionId));

        // When
        String submissionId =
                acknowledgementService.getLatestSubmissionIdByTaxReturnIdPreferringAcceptedSubmission(taxReturnId);

        // Then
        verify(taxReturnSubmissionRepository, times(1)).getLatestAcceptedSubmissionIdForTaxReturnId(taxReturnId);
        verify(taxReturnSubmissionRepository, times(0)).getLatestSubmissionIdByTaxReturnId(taxReturnId);
        assertEquals(latestAcceptedSubmissionId, submissionId);
    }

    @Test
    void getLatestAcceptedSubmissionIdOfParentTaxReturn_returnsEmptyIfNoAcceptedSubmissionFound() {
        // Given
        var requestedSubmissionId = "requestedSubmissionId";

        when(taxReturnSubmissionRepository.getLatestAcceptedSubmissionIdOfParentTaxReturn(requestedSubmissionId))
                .thenReturn(Optional.empty());

        // When
        Optional<String> latestAcceptedSubmissionId =
                acknowledgementService.getLatestAcceptedSubmissionIdOfParentTaxReturn(requestedSubmissionId);

        // Then
        verify(taxReturnSubmissionRepository, times(1))
                .getLatestAcceptedSubmissionIdOfParentTaxReturn(requestedSubmissionId);
        assertTrue(latestAcceptedSubmissionId.isEmpty());
    }

    @Test
    void getLatestAcceptedSubmissionIdOfParentTaxReturn_returnsTheAcceptedSubmissionId() {
        // Given
        var requestedSubmissionId = "requestedSubmissionId";
        var acceptedSubmissionId = "acceptedSubmissionId";

        when(taxReturnSubmissionRepository.getLatestAcceptedSubmissionIdOfParentTaxReturn(requestedSubmissionId))
                .thenReturn(Optional.of(acceptedSubmissionId));

        // When
        Optional<String> latestAcceptedSubmissionId =
                acknowledgementService.getLatestAcceptedSubmissionIdOfParentTaxReturn(requestedSubmissionId);

        // Then
        verify(taxReturnSubmissionRepository, times(1))
                .getLatestAcceptedSubmissionIdOfParentTaxReturn(requestedSubmissionId);
        assertTrue(latestAcceptedSubmissionId.isPresent());
        assertEquals(acceptedSubmissionId, latestAcceptedSubmissionId.get());
    }
}
