package gov.irs.directfile.api.io.documentstore;

import java.io.*;

import gov.irs.directfile.api.io.IIOLocation;
import gov.irs.directfile.api.io.IOLocationException;

public class DocumentStoreIOLocation implements IIOLocation {
    private final S3StorageService s3StorageService;

    public DocumentStoreIOLocation(S3StorageService s3StorageService) {
        this.s3StorageService = s3StorageService;
    }

    @Override
    public InputStream read(String location) throws IOLocationException {
        try {
            return s3StorageService.download(location);
        } catch (IOException e) {
            throw new IOLocationException(String.format("Could not access file at location: %s", location), e);
        }
    }

    @Override
    public void write(String location, InputStream payloadStream) throws IOLocationException {
        try {
            s3StorageService.write(location, payloadStream);
        } catch (IOException e) {
            throw new IOLocationException(
                    String.format("Failed to write document to document store at %s", location), e);
        }
    }

    @Override
    public void delete(String location) throws IOLocationException {
        throw new IOLocationException("Delete not supported for this IO type.");
    }
}
