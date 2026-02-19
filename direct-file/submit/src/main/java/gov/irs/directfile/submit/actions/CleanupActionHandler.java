package gov.irs.directfile.submit.actions;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.submit.actions.results.CleanupActionResult;
import gov.irs.directfile.submit.command.CleanupAction;
import gov.irs.directfile.submit.domain.DocumentStoreResource;
import gov.irs.directfile.submit.domain.SubmissionBatch;
import gov.irs.directfile.submit.service.interfaces.ISynchronousDocumentStoreService;

@Slf4j
@SuppressFBWarnings(
        value = {"NM_METHOD_NAMING_CONVENTION"},
        justification = "Initial Spotbugs Setup")
@Service
public class CleanupActionHandler {

    private final ActionContext actionContext;
    private final ISynchronousDocumentStoreService documentStoreService;

    public CleanupActionHandler(ActionContext actionContext, ISynchronousDocumentStoreService documentStoreService) {
        this.actionContext = actionContext;
        this.documentStoreService = documentStoreService;
    }

    public void cleanupBatch(SubmissionBatch submissionBatch) throws ActionException {
        cleanDirectory(Path.of(actionContext.getConfig().getDirectories().getToProcess()));
        cleanDirectory(Path.of(actionContext.getConfig().getDirectories().getProcessed()));
        cleanDirectory(Path.of(actionContext.getConfig().getDirectories().getToBatch()));
        cleanDirectory(Path.of(actionContext.getConfig().getDirectories().getBatched()));
        cleanDocumentStore(submissionBatch);
    }

    public CleanupActionResult handleAction(CleanupAction cleanupAction) throws ActionException {
        this.cleanupBatch(cleanupAction.getSubmissionBatch());
        return new CleanupActionResult();
    }

    void cleanDirectory(Path path) {
        // check if the path exists
        if (!Files.exists(path)) {
            log.info("Path does not exist: {}", path);
            return;
        }

        // clear the files so the directories can be deleted.
        try (Stream<Path> walk = Files.walk(path)) {
            walk.filter(Files::isRegularFile).forEach(x -> {
                try {
                    Files.deleteIfExists(x);
                } catch (IOException e) {
                    log.error("Error deleting file: {}", x, e);
                }
            });
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // same thing, but with the directories
        try (Stream<Path> walk = Files.walk(path)) {
            walk.filter(Files::isDirectory).forEach(x -> {
                if (x.compareTo(path) != 0) {
                    try {
                        Files.delete(x);
                    } catch (IOException e) {
                        log.error("Error deleting directory: {}", x, e);
                    }
                }
            });
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    void cleanDocumentStore(SubmissionBatch batch) {
        List<String> allKeys = documentStoreService.getObjectKeys(batch.path()).stream()
                .map(DocumentStoreResource::getFullLocation)
                .toList();

        // 2. Do a nested for loop in batches of 1000 to delete the objects from s3
        int maxObjectsPerRequest = 1000;
        for (int i = 0; i < allKeys.size(); i += maxObjectsPerRequest) {
            List<String> partition = allKeys.subList(i, Math.min(i + maxObjectsPerRequest, allKeys.size()));
            documentStoreService.deleteObjects(partition);
        }
    }
}
