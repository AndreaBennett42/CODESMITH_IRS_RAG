package gov.irs.directfile.api.dispatch;

import lombok.AllArgsConstructor;

@AllArgsConstructor
public class DispatchContext {

    public String pathToSubmissionXml;
    public String pathToManifestXml;
    public String pathToUserContext;
    public String submissionId;
}
