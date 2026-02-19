package gov.irs.directfile.api.io.resource;

import java.io.IOException;
import java.io.InputStream;

import org.springframework.core.io.ClassPathResource;

import gov.irs.directfile.api.io.IIOLocation;
import gov.irs.directfile.api.io.IOLocationException;

public class ResourceIOLocation implements IIOLocation {
    @Override
    public InputStream read(String location) throws IOLocationException {
        try {
            return new ClassPathResource(location).getInputStream();
        } catch (IOException e) {
            throw new IOLocationException(String.format("Could not access resource at location: %s", location), e);
        }
    }

    @Override
    public void write(String location, InputStream payloadStream) throws IOLocationException {}

    @Override
    public void delete(String location) throws IOLocationException {}
}
