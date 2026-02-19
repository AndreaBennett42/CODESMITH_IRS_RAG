package gov.irs.directfile.submit.actions;

public interface IAction<T> {
    T Act(ActionContext context) throws ActionException;
}
