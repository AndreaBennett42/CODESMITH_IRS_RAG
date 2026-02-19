package gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.pdfBackfill;

import java.io.IOException;
import java.io.InputStream;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Limit;
import org.springframework.data.domain.ScrollPosition;
import org.springframework.data.domain.Window;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import gov.irs.directfile.api.io.documentstore.S3StorageService;
import gov.irs.directfile.api.io.storagelocations.StorageLocationBuilder;
import gov.irs.directfile.api.pdf.PdfCreationException;
import gov.irs.directfile.api.pdf.PdfLanguages;
import gov.irs.directfile.api.pdf.PdfService;
import gov.irs.directfile.api.taxreturn.TaxReturnRepository;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.taxreturn.submissions.handlers.s3eventnotification.S3NotificationEventHandler;

@Service
@Transactional
@Slf4j
public class PDFBackfillToS3Handler implements S3NotificationEventHandler {
    private final TaxReturnRepository taxReturnRepository;
    private final PdfService pdfService;
    private final S3StorageService s3StorageService;

    private static final int MAX_TAX_RETURNS_PER_PAGE = 250;
    private static final int MIN_TAX_RETURNS_PER_PAGE = 1;

    private static final int TAX_RETURN_BATCH_SIZE = 50;
    private static final String BASE_NAME = "taxreturn";
    private final ExecutorService executorService;

    public PDFBackfillToS3Handler(
            TaxReturnRepository taxReturnRepository, PdfService pdfService, S3StorageService s3StorageService) {
        this.taxReturnRepository = taxReturnRepository;
        this.pdfService = pdfService;
        this.s3StorageService = s3StorageService;
        /** Note: I wasn't sure how many threads to use in this thread pool. So I used a formula from this article:
         * https://engineering.zalando.com/posts/2019/04/how-to-set-an-ideal-thread-pool-size.html
         *
         * Number of threads = Number of Available Cores * (1 + Wait time / Service time)
         * Waiting time:  is the time spent waiting for IO bound tasks to complete, say waiting for HTTP response from
         * remote service.
         * Service Time:  is the time spent being busy, say processing the HTTP response, marshaling/unmarshaling, any
         * other transformations etc.
         *
         * Wait Time: Assume S3 upload for both files takes 3 seconds -> 3000ms (S3 is way faster, but just chose a round number)
         * Service Time: We don't do anything, but I'll say we have 1000ms of processing time
         */
        int totalThreads = Runtime.getRuntime().availableProcessors() * (1 + (3000 / 1000));
        log.info("Running executor service with {} threads.", totalThreads);
        this.executorService = Executors.newFixedThreadPool(totalThreads);
    }

    @PreDestroy
    public void teardown() {
        executorService.shutdown();
    }

    @Override
    public void handleNotificationEvent(JsonNode payload) {
        try {
            Date startDate = new SimpleDateFormat("yyyy-MM-dd", Locale.US)
                    .parse(payload.get("startDate").asText());
            Date endDate = new SimpleDateFormat("yyyy-MM-dd", Locale.US)
                    .parse(payload.get("endDate").asText());

            int resultsPerPage = payload.get("resultsPerPage").asInt();
            int taxYear = payload.get("taxYear").asInt();
            /* Note: Tax Return Ids is an optional field. If not ids are provided, the handler will just page through the database based on the [startDate, endDate] range */
            if (payload.has("taxReturnIds")) {
                Iterator<JsonNode> idsArr = payload.get("taxReturnIds").elements();
                List<UUID> uuids = new ArrayList<>();

                idsArr.forEachRemaining(taxReturnUUIDJson -> {
                    uuids.add(UUID.fromString(taxReturnUUIDJson.asText()));
                });
                new Thread(() -> generatePDFsFromTaxReturnIdList(uuids)).start();
            } else {
                int queryLimit = Math.min(Math.max(MIN_TAX_RETURNS_PER_PAGE, resultsPerPage), MAX_TAX_RETURNS_PER_PAGE);
                new Thread(() -> generatePDFsForS3(queryLimit, startDate, endDate, taxYear)).start();
            }

        } catch (ParseException e) {
            log.error("Unable to parse json");
        }
    }

    public void generatePDFsFromTaxReturnIdList(List<UUID> taxReturnIds) {
        // 1. Partition list into buckets of 50
        List<List<UUID>> partitions = partitionList(taxReturnIds, TAX_RETURN_BATCH_SIZE);
        List<UUID> failedTaxReturnsUUIDs = new ArrayList<>();
        List<UUID> successfulTaxReturnUUIDs = new ArrayList<>();

        for (List<UUID> partition : partitions) {
            List<TaxReturn> taxReturns = taxReturnRepository.findAllByTaxReturnIds(partition);
            List<Callable<Void>> tasks = new ArrayList<>();

            try {
                for (TaxReturn taxReturn : taxReturns) {
                    tasks.add(uploadPdfTask(taxReturn, failedTaxReturnsUUIDs, successfulTaxReturnUUIDs));
                }
                executorService.invokeAll(tasks);
            } catch (InterruptedException e) {
                log.error("Upload interuppted for partition with ids: {}", partition);
                failedTaxReturnsUUIDs.addAll(
                        taxReturns.stream().map(TaxReturn::getId).toList());
            }
        }
        log.info("Completed persistence of all pdfs to Amazon s3. " + " Total Tax Returns Processed: "
                + (successfulTaxReturnUUIDs.size() + failedTaxReturnsUUIDs.size()) + " "
                + " Successful Tax Returns Processed: " + successfulTaxReturnUUIDs.size() + " "
                + " Failed Tax Returns Processed: "
                + failedTaxReturnsUUIDs.size());
        log.info("Unable to persist PDFs for the following tax returns: {} ", failedTaxReturnsUUIDs);
    }

    public void generatePDFsForS3(int limit, Date startDate, Date endDate, int taxYear) {
        // Based on Spring Docs for Scrolling:
        // https://docs.spring.io/spring-data/jpa/reference/jpa/query-methods.html#repositories.scrolling
        Limit resultsLimit = Limit.of(limit);
        Window<TaxReturn> taxReturns = taxReturnRepository.findByTaxYearAndCreatedAtBetweenOrderByCreatedAtAsc(
                resultsLimit, ScrollPosition.offset(), taxYear, startDate, endDate);

        // Based on Spring Docs for scrolling through Window:
        // https://docs.spring.io/spring-data/jpa/reference/data-commons/repositories/scrolling.html
        log.info(
                "Generating PDFs for time range - start date: {} , end date: {} with page size: {}",
                startDate,
                endDate,
                limit);
        List<UUID> failedTaxReturnUUIDs = new ArrayList<>();
        List<UUID> successfulTaxReturnUUIDs = new ArrayList<>();

        while (!taxReturns.isEmpty()) {
            try {
                List<Callable<Void>> tasks = new ArrayList<>();
                for (TaxReturn taxReturn : taxReturns) {
                    if (taxReturn.hasBeenSubmittedAtLeastOnce()) {
                        tasks.add(uploadPdfTask(taxReturn, failedTaxReturnUUIDs, successfulTaxReturnUUIDs));
                    }
                }
                executorService.invokeAll(tasks);
            } catch (InterruptedException e) {
                log.error("Error processing tax returns in the batch.");
                failedTaxReturnUUIDs.addAll(
                        taxReturns.stream().map(TaxReturn::getId).toList());
            }

            taxReturns = taxReturnRepository.findByTaxYearAndCreatedAtBetweenOrderByCreatedAtAsc(
                    resultsLimit, taxReturns.positionAt(taxReturns.size() - 1), taxYear, startDate, endDate);
        }

        log.info(
                "Completed PDF Backfill for time range - start date: {} , end date: {}. "
                        + "Successfully Processed Returns Count: {}   Processing Failure Count: {}",
                startDate,
                endDate,
                successfulTaxReturnUUIDs.size(),
                failedTaxReturnUUIDs.size());
        log.info(
                "Unable to persist PDFs for the following tax returns for time range - start date: {} , end date: {} : {} ",
                startDate,
                endDate,
                failedTaxReturnUUIDs);
    }

    private Callable<Void> uploadPdfTask(
            TaxReturn taxReturn, List<UUID> failedTaxReturns, List<UUID> successTaxReturns) {
        return () -> {
            try (InputStream englishTaxReturnPdf =
                            pdfService.getTaxReturn(PdfLanguages.EN.getPdfLanguage(), taxReturn, false);
                    InputStream spanishTaxReturnPdf =
                            pdfService.getTaxReturn(PdfLanguages.ES.getPdfLanguage(), taxReturn, false)) {

                String englishPdfLocation = StorageLocationBuilder.getTaxReturnDocumentLocation(
                        taxReturn.getTaxYear(), taxReturn.getId(), BASE_NAME, PdfLanguages.EN.getCode());
                String spanishPdfLocation = StorageLocationBuilder.getTaxReturnDocumentLocation(
                        taxReturn.getTaxYear(), taxReturn.getId(), BASE_NAME, PdfLanguages.ES.getCode());
                s3StorageService.write(englishPdfLocation, englishTaxReturnPdf, null);
                log.info("English pdf for tax return: {} written to {}", taxReturn.getId(), englishPdfLocation);
                s3StorageService.write(spanishPdfLocation, spanishTaxReturnPdf, null);
                log.info("Spanish pdf for tax return: {} written to {}", taxReturn.getId(), spanishPdfLocation);

                log.info(
                        "Published PDFs for tax return {} created at : {}",
                        taxReturn.getId(),
                        taxReturn.getCreatedAt());
            } catch (PdfCreationException e) {
                log.error(
                        "Unable to generate PDF for tax return: {} created_at: {}",
                        taxReturn.getId(),
                        taxReturn.getCreatedAt(),
                        e);
                failedTaxReturns.add(taxReturn.getId());
            } catch (IOException e) {
                log.error(
                        "Failed to write PDF to s3 for tax return: {} created_at: {} ",
                        taxReturn.getId(),
                        taxReturn.getCreatedAt(),
                        e);
                failedTaxReturns.add(taxReturn.getId());
            }
            successTaxReturns.add(taxReturn.getId());
            return null;
        };
    }

    private List<List<UUID>> partitionList(List<UUID> taxReturnUUIDs, int batchSize) {
        List<List<UUID>> result = new ArrayList<>();
        for (int i = 0; i < taxReturnUUIDs.size(); i += batchSize) {
            List<UUID> partition = taxReturnUUIDs.subList(i, Math.min(i + batchSize, taxReturnUUIDs.size()));
            result.add(partition);
        }
        return result;
    }
}
