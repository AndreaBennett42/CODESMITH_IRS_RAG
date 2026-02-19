package gov.irs.directfile.submit.actions.exception;

import lombok.Getter;

import gov.irs.directfile.submit.actions.ActionException;

@Getter
public class CreateArchiveActionException extends ActionException {

    public CreateArchiveActionException(Throwable e) {
        super(e);
    }
}
