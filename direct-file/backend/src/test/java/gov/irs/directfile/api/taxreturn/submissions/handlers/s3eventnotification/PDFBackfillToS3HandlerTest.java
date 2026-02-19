package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;
import java.util.UUID;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.IntNode;
import com.google.common.base.Charsets;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.TestEntityManager;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Bean;
import software.amazon.awssdk.core.ResponseBytes;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;

import gov.irs.directfile.api.config.S3ConfigurationProperties;
import gov.irs.directfile.api.config.S3ConfigurationProperties.S3;
import gov.irs.directfile.api.config.identity.IdentityAttributes;
import gov.irs.directfile.api.config.identity.IdentitySupplier;
import gov.irs.directfile.api.io.documentstore.S3StorageService;
import gov.irs.directfile.api.pdf.PdfService;
import gov.irs.directfile.api.taxreturn.TaxReturnRepository;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.taxreturn.submissions.S3NotificationEventRouter;
import gov.irs.directfile.api.taxreturn.submissions.S3NotificationEventService;
import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.pdfBackfill.PDFBackfillToS3Handler;
import gov.irs.directfile.api.user.models.User;
import gov.irs.directfile.api.util.base.BaseRepositoryTest;
import gov.irs.directfile.models.FactTypeWithItem;
import gov.irs.directfile.models.encryption.DataEncryptDecrypt;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class PDFBackfillToS3HandlerTest extends BaseRepositoryTest {
    ObjectMapper mapper = new ObjectMapper();

    @Autowired
    private TestEntityManager entityManager;

    @Autowired
    TaxReturnRepository taxReturnRepository;

    PDFBackfillToS3Handler pdfBackfillToS3Handler;

    S3ConfigurationProperties s3ConfigurationProperties = new S3ConfigurationProperties(
            null, null, new S3(null, null, 0, null, null, "some-bucket", "some-operations-jobs-bucket", "dev"));

    @TestConfiguration
    public static class TestConfig {
        // For autowiring into TaxReturnEntityListener
        @Bean
        public ObjectMapper getObjectMapper() {
            return new ObjectMapper();
        }
    }

    @MockBean
    private DataEncryptDecrypt dataEncryptDecrypt;

    @MockBean
    private IdentitySupplier mockIdentitySupplier;

    @Autowired
    private TaxReturnRepository taxReturnRepo;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    PdfService pdfService;

    @MockBean
    S3StorageService s3StorageService;

    @Mock(name = "s3WithoutEncryption")
    S3Client mockS3Client;

    S3NotificationEventRouter notificationEventRouter;

    S3NotificationEventService s3NotificationEventService;

    String backfillPDFsByDateRangeJson =
            "{\"key\":\"backfill_pdfs\",\"payload\" : {\"startDate\" : \"2024-01-01\",\"endDate\" : \"2024-06-30\",\"taxYear\" : \"2023\", \"resultsPerPage\":\"10\"}}";
    String backfillPDFsByTaxReturnIdsJson =
            "{\"key\":\"backfill_pdfs\", \"payload\" : {\"startDate\" : \"2024-01-01\", \"endDate\" : \"2024-06-30\", \"taxYear\" : \"2023\",  \"resultsPerPage\":\"1\"}}";

    JsonNode backfillByDateRangePayload;
    JsonNode backfillByTaxReturnIdPayload;

    @BeforeEach
    void configure() throws Exception {
        doReturn(new IdentityAttributes(UUID.randomUUID(), UUID.randomUUID(), "email@example.com", "123456789"))
                .when(mockIdentitySupplier)
                .get();

        pdfBackfillToS3Handler = new PDFBackfillToS3Handler(taxReturnRepository, pdfService, s3StorageService);
        notificationEventRouter = new S3NotificationEventRouter(null, null, pdfBackfillToS3Handler, null, null);
        s3NotificationEventService =
                new S3NotificationEventService(notificationEventRouter, mockS3Client, s3ConfigurationProperties);

        GetObjectResponse getObjectResponse = GetObjectResponse.builder().build();
        ResponseBytes<GetObjectResponse> responseBytes =
                ResponseBytes.fromByteArray(getObjectResponse, backfillPDFsByDateRangeJson.getBytes());
        when(mockS3Client.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(responseBytes);
        backfillByDateRangePayload = s3NotificationEventService
                .loadObjectFromS3("post_submission_error_backfill_pdfs.json", "some-bucket")
                .get("payload");
        backfillByTaxReturnIdPayload = mapper.readTree(backfillPDFsByTaxReturnIdsJson.getBytes(Charsets.UTF_8))
                .get("payload");
    }

    // @Test
    public void itGeneratesPDFForEachUser() throws Exception {
        // given user has no returns
        User user = new User(UUID.fromString("738fc2dd-88f9-4b5c-ace9-c602509ba161"));
        user = entityManager.persist(user);

        // given user has one return
        Date march20_2024 = new SimpleDateFormat("yyyy-MM-dd").parse("2024-03-20");
        Date nov20_2024 = new SimpleDateFormat("yyyy-MM-dd").parse("2024-11-20");

        TaxReturn taxReturn = TaxReturn.testObjectFactoryNoId(march20_2024);
        taxReturn.setFacts(Map.of("testA", new FactTypeWithItem("typeA", new IntNode(24))));
        byte[] factsBytes = objectMapper.writeValueAsBytes(taxReturn.getFacts());
        when(dataEncryptDecrypt.encrypt(eq(factsBytes), anyMap())).thenReturn(factsBytes);
        when(dataEncryptDecrypt.decrypt(factsBytes)).thenReturn(factsBytes);
        user.addTaxReturn(taxReturn);
        TaxReturn t = entityManager.persist(taxReturn);

        User user2 = new User(UUID.randomUUID());
        user2 = entityManager.persist(user2);

        // given user has one return
        TaxReturn taxReturn2 = TaxReturn.testObjectFactoryNoId(march20_2024);
        taxReturn.setFacts(Map.of("testA", new FactTypeWithItem("typeA", new IntNode(24))));
        byte[] factsBytes2 = objectMapper.writeValueAsBytes(taxReturn2.getFacts());
        when(dataEncryptDecrypt.encrypt(eq(factsBytes2), anyMap())).thenReturn(factsBytes2);
        when(dataEncryptDecrypt.decrypt(factsBytes2)).thenReturn(factsBytes2);
        user2.addTaxReturn(taxReturn2);
        entityManager.persist(taxReturn2);

        Date startDate = new SimpleDateFormat("yyyy-MM-dd").parse("2024-01-01");
        Date endDate = new SimpleDateFormat("yyyy-MM-dd").parse("2024-06-30");

        int taxYear = 2023;
        pdfBackfillToS3Handler.generatePDFsForS3(1, startDate, endDate, taxYear);

        verify(pdfService, times(4)).getTaxReturn(any(), any(), eq(false));
    }

    // @Test
    /**
     * Processing PDFs happens in a separate thread, so I've included a timer of 2 seconds,
     * which I expect will give the thread enough time to run.
     *
     * */
    public void itParsesJSONNodeAndGeneratesPDFs() throws Exception {
        // given user has no returns
        User user = new User(UUID.fromString("738fc2dd-88f9-4b5c-ace9-c602509ba161"));
        user = entityManager.persist(user);

        // given user has one return
        TaxReturn taxReturn = TaxReturn.testObjectFactoryNoId();
        taxReturn.setFacts(Map.of("testA", new FactTypeWithItem("typeA", new IntNode(24))));

        byte[] factsBytes = objectMapper.writeValueAsBytes(taxReturn.getFacts());
        when(dataEncryptDecrypt.encrypt(eq(factsBytes), anyMap())).thenReturn(factsBytes);
        when(dataEncryptDecrypt.decrypt(factsBytes)).thenReturn(factsBytes);
        user.addTaxReturn(taxReturn);
        entityManager.persist(taxReturn);

        User user2 = new User(UUID.randomUUID());
        user2 = entityManager.persist(user2);

        // given user has one return
        TaxReturn taxReturn2 = TaxReturn.testObjectFactoryNoId();
        taxReturn.setFacts(Map.of("testA", new FactTypeWithItem("typeA", new IntNode(24))));
        byte[] factsBytes2 = objectMapper.writeValueAsBytes(taxReturn2.getFacts());
        when(dataEncryptDecrypt.encrypt(eq(factsBytes2), anyMap())).thenReturn(factsBytes2);
        when(dataEncryptDecrypt.decrypt(factsBytes2)).thenReturn(factsBytes2);
        user2.addTaxReturn(taxReturn2);
        entityManager.persist(taxReturn2);

        pdfBackfillToS3Handler.handleNotificationEvent(backfillByDateRangePayload);

        verify(pdfService, times(4)).getTaxReturn(any(), any(), eq(false));
    }
}
