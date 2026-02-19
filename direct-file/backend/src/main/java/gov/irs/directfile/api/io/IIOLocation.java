package gov.irs.directfile.api.io;

import java.io.InputStream;

public interface IIOLocation {
    // note for future devs:
    // these may require some kind of options some day.
    InputStream read(String location) throws IOLocationException;

    void write(String location, InputStream payloadStream) throws IOLocationException;

    void delete(String location) throws IOLocationException;
}
