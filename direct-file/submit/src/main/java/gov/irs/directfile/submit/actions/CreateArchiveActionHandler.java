package gov.irs.directfile.submit.actions;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import javax.xml.datatype.XMLGregorianCalendar;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.mef.exception.ToolkitException;
import gov.irs.mef.inputcomposition.*;

import gov.irs.directfile.submit.actions.exception.CreateArchiveActionException;
import gov.irs.directfile.submit.actions.results.CreateArchiveActionResult;
import gov.irs.directfile.submit.command.CreateArchiveAction;
import gov.irs.directfile.submit.domain.SubmissionArchiveContainer;
import gov.irs.directfile.submit.domain.UserContextData;
import gov.irs.directfile.submit.service.LocalWriteUtilityService;
import gov.irs.directfile.submit.service.interfaces.ISynchronousDocumentStoreService;

@Slf4j
@SuppressFBWarnings(
        value = {"NM_METHOD_NAMING_CONVENTION"},
        justification = "Initial SpotBugs Setup")
@Service
public class CreateArchiveActionHandler {
    private final ISynchronousDocumentStoreService storageService;

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final ActionContext actionContext;

    public CreateArchiveActionHandler(ActionContext actionContext, ISynchronousDocumentStoreService storageService) {
        this.actionContext = actionContext;
        this.storageService = storageService;
    }

    public CreateArchiveActionResult handleCommand(CreateArchiveAction createArchiveActionCommand)
            throws ActionException {
        ArrayList<SubmissionArchiveContainer> archives = new ArrayList<>();
        List<String> userSubmissions = storageService.getSubFolders(
                createArchiveActionCommand.getSubmissionBatch().path());

        if (!userSubmissions.isEmpty()) {
            log.info(
                    "Creating archives for xmls with manifests for batch {}",
                    createArchiveActionCommand.getSubmissionBatch());
        }
        for (String submissionObjectKey : userSubmissions) {
            SubmissionArchive archive = null;

            try {
                // 1a. Pull down the UserContext b/c we need it for everything else
                String userContextDataObjectKey = submissionObjectKey + "userContext.json";
                String userContextJsonString = storageService.getObjectAsString(userContextDataObjectKey);
                UserContextData UserContext = objectMapper.readValue(userContextJsonString, UserContextData.class);

                // 1b. Pull down submission.xml and write to disk so later actions can access it
                String submissionXMLKey = submissionObjectKey + "submission.xml";
                String submissionXMLString = storageService.getObjectAsString(submissionXMLKey);
                Path submissionXml = LocalWriteUtilityService.writeXmlToDisk(
                        submissionXMLString,
                        UserContext.getSubmissionId(),
                        actionContext.getConfig().getDirectories().getProcessed(),
                        "submission");

                // 1c. Pull down manifest.xml and write to disk so later actions can access it
                String manifestXMLKey = submissionObjectKey + "manifest.xml";
                String manifestXMString = storageService.getObjectAsString(manifestXMLKey);
                Path manifestXml = LocalWriteUtilityService.writeXmlToDisk(
                        manifestXMString,
                        UserContext.getSubmissionId(),
                        actionContext.getConfig().getDirectories().getProcessed(),
                        UserContext.getSubmissionId() + ".manifest");

                // 2. Create an archive for the submission
                archive = SubmissionBuilder.createIRSSubmissionArchive(
                        UserContext.getSubmissionId(),
                        new SubmissionManifest(manifestXml.toFile()),
                        new SubmissionXML(submissionXml.toFile()),
                        new BinaryAttachment
                                [] {}, // Passing empty array for Binary Attachments because we don't use them
                        actionContext.getConfig().getDirectories().getToBatch());

                // 3. Create a timestamped archive so that users in PT can file in their time
                XMLGregorianCalendar calendar;
                try {
                    calendar = LocalWriteUtilityService.CreateGregorianDateFromString(UserContext.getSignDate());
                } catch (Exception ex) {
                    log.error("Attempted to create a timestamped archive without a valid timestamp.  Using today.");
                    calendar = LocalWriteUtilityService.Today();
                }
                var timestampedArchive =
                        SubmissionBuilder.createPostmarkedSubmissionArchive(archive, calendar.toGregorianCalendar());

                archives.add(new SubmissionArchiveContainer(UserContext, timestampedArchive));
            } catch (ToolkitException | IOException e) {
                throw new CreateArchiveActionException(e);
            }
        }

        return new CreateArchiveActionResult(createArchiveActionCommand.getSubmissionBatch(), archives);
    }
}
