package gov.irs.directfile.submit.command;

import lombok.AllArgsConstructor;
import lombok.Getter;

import gov.irs.directfile.submit.actions.results.BundleArchivesActionResult;

@Getter
@AllArgsConstructor
public class SubmitBundleAction extends Action {

    private final BundleArchivesActionResult bundleArchivesActionResult;

    @Override
    public ActionType getType() {
        return ActionType.SUBMIT_BUNDLE;
    }
}
