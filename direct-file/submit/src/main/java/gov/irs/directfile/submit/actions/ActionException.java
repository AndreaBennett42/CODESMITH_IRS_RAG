package gov.irs.directfile.submit.actions;

public class ActionException extends Exception {

    public ActionException(String message) {
        super(message);
    }

    public ActionException(String message, Throwable cause) {
        super(message, cause);
    }

    public ActionException(Throwable e) {
        super(e);
    }
}
