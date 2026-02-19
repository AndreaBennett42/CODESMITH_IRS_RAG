package gov.irs.directfile.api.taxreturn.submissions;

import java.io.Serializable;
import java.util.List;

import gov.irs.directfile.models.email.HtmlTemplate;

public record SubmissionStatusesMessage(HtmlTemplate status, List<String> submissionIds) implements Serializable {}
