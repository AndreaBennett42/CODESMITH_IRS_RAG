package gov.irs.directfile.submit.command;

import lombok.AllArgsConstructor;
import lombok.Getter;

import gov.irs.directfile.submit.actions.results.CreateArchiveActionResult;

@Getter
@AllArgsConstructor
public class BundleArchiveAction extends Action {

    private final CreateArchiveActionResult createArchiveActionResult;

    @Override
    public ActionType getType() {
        return ActionType.BUNDLE_ARCHIVE;
    }
}
