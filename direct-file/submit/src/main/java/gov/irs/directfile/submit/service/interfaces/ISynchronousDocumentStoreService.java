package gov.irs.directfile.submit.service.interfaces;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Optional;

import gov.irs.directfile.submit.domain.DocumentStoreResource;

public interface ISynchronousDocumentStoreService extends IService {
    String write(String objectKey, InputStream payloadStream) throws IOException;

    String write(String objectKey, String content);

    Optional<String> getMostRecentFolderForPrefix(String prefix);

    List<String> getSubFolders(String objectKey);

    Optional<DocumentStoreResource> getLeastRecentModifiedResourceForPrefix(String s);

    String getObjectAsString(String objectKey) throws IOException;

    List<DocumentStoreResource> getObjectKeys(String prefix);

    void deleteObjects(List<String> keys);

    void copyObject(DocumentStoreResource documentStoreResource, String destinationKey);
}
