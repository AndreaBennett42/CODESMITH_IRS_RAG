package gov.irs.directfile.submit.domain;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;

import gov.irs.mef.inputcomposition.BinaryAttachment;
import gov.irs.mef.inputcomposition.SubmissionManifest;
import gov.irs.mef.inputcomposition.SubmissionXML;

@SuppressFBWarnings(
        value = {"URF_UNREAD_PUBLIC_OR_PROTECTED_FIELD"},
        justification = "Initial SpotBugs Setup")
@SuppressWarnings("PMD.ArrayIsStoredDirectly")
public class SubmissionData {
    public SubmissionData(
            UserContextData userContext,
            SubmissionManifest manifest,
            SubmissionXML xml,
            BinaryAttachment[] binaryObjects) {
        UserContext = userContext;
        Manifest = manifest;
        Xml = xml;
        BinaryObjects = binaryObjects;
    }

    public UserContextData UserContext;
    public SubmissionManifest Manifest;
    public SubmissionXML Xml;
    public BinaryAttachment[] BinaryObjects;
}
