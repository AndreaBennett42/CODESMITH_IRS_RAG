package gov.irs.directfile.api.io.directory;

import java.io.*;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.compress.utils.IOUtils;

import gov.irs.directfile.api.io.IIOLocation;
import gov.irs.directfile.api.io.IOLocationException;

@Slf4j
@SuppressFBWarnings(
        value = {"PATH_TRAVERSAL_IN"},
        justification = "This method is only called by trusted classes, so we consider this a false positive.")
public class DirectoryIOLocation implements IIOLocation {

    @Override
    public InputStream read(String location) throws IOLocationException {
        try {
            return new FileInputStream(location);
        } catch (FileNotFoundException e) {
            throw new IOLocationException(String.format("Could not access file at location: %s", location), e);
        }
    }

    @Override
    public void write(String location, InputStream payloadStream) throws IOLocationException {
        File f = new File(location);
        try (FileOutputStream fileOut = new FileOutputStream(f); ) {
            IOUtils.copy(payloadStream, fileOut);
        } catch (FileNotFoundException e) {
            throw new IOLocationException("Could not create a file at provided path", e);
        } catch (IOException e) {
            throw new IOLocationException("Could not copy from provided stream", e);
        }
    }

    @Override
    public void delete(String location) throws IOLocationException {
        try {
            File f = new File(location);
            if (f.exists() && f.isFile()) {
                boolean deleteResult = f.delete();
                if (!deleteResult) {
                    log.warn("Failed to remove file: {}", location);
                }
            }
        } catch (Exception e) {
            throw new IOLocationException(String.format("Could not delete file at location %s", location), e);
        }
    }
}
