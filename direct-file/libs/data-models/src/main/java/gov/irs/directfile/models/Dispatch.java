package gov.irs.directfile.models;

import java.util.*;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Dispatch {
    private UUID id = UUID.randomUUID();
    private UUID userId;
    private UUID taxReturnId;
    private String pathToManifest;
    private String pathToUserContext;
    private String pathToSubmission;
    private String mefSubmissionId;

    public Dispatch() {}

    public Dispatch(
            UUID userId,
            UUID taxReturnId,
            String pathToManifest,
            String pathToUserContext,
            String pathToSubmission,
            String mefSubmissionId) {
        this.userId = userId;
        this.taxReturnId = taxReturnId;
        this.pathToManifest = pathToManifest;
        this.pathToUserContext = pathToUserContext;
        this.pathToSubmission = pathToSubmission;
        this.mefSubmissionId = mefSubmissionId;
    }

    public static Dispatch testObjectFactory() {
        Dispatch dispatch = new Dispatch();
        dispatch.id = UUID.randomUUID();
        dispatch.userId = UUID.randomUUID();
        dispatch.taxReturnId = UUID.randomUUID();
        dispatch.mefSubmissionId = "12345";
        dispatch.pathToManifest = "";
        dispatch.pathToUserContext = "";
        dispatch.pathToSubmission = "";
        return dispatch;
    }
}
