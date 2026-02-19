package gov.irs.directfile.submit.actions.results;

import lombok.AllArgsConstructor;
import lombok.Getter;

import gov.irs.directfile.submit.domain.BundledArchives;
import gov.irs.directfile.submit.domain.SubmissionBatch;

@Getter
@AllArgsConstructor
public class BundleArchivesActionResult {
    private SubmissionBatch batch;
    private BundledArchives bundledArchives;
}
