package gov.irs.directfile.api.io;

import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.api.errors.DefaultCaseException;
import gov.irs.directfile.api.io.directory.DirectoryIOLocation;
import gov.irs.directfile.api.io.documentstore.DocumentStoreIOLocation;
import gov.irs.directfile.api.io.documentstore.S3StorageService;
import gov.irs.directfile.api.io.memory.MemoryIOLocation;
import gov.irs.directfile.api.io.resource.ResourceIOLocation;

@Service
@Slf4j
public class IOLocationService {
    private final S3StorageService s3StorageService;

    public enum ConfiguredLocations {
        classpath,
        directory,
        memory,
        documentstore
    }

    public static ConfiguredLocations getConfiguredLocationType(String name) throws DefaultCaseException {
        switch (name.trim().toLowerCase()) {
            case "classpath":
                return ConfiguredLocations.classpath;
            case "directory":
                return ConfiguredLocations.directory;
            case "memory":
                return ConfiguredLocations.memory;
            case "documentstore":
                return ConfiguredLocations.documentstore;
            default:
                throw new DefaultCaseException(String.format("Location %s is not configured", name));
        }
    }
    // the intent of the location service is to provide the system
    // with a transparent method for getting whatever data is required.

    // Modules themselves shouldn't have to know where the information they
    // want is stored.  They should be configured to use some storage and
    // the rest should be provided for them through a generic interface
    private final Map<ConfiguredLocations, IIOLocation> locations;

    public IOLocationService(S3StorageService s3StorageService) {
        this.s3StorageService = s3StorageService;
        this.locations = new HashMap<>();

        // TODO: these registrations could go elsewhere when we have more (S3, kafka, etc)
        this.registerLocationType(ConfiguredLocations.classpath, new ResourceIOLocation());
        this.registerLocationType(ConfiguredLocations.directory, new DirectoryIOLocation());
        this.registerLocationType(ConfiguredLocations.memory, new MemoryIOLocation());
        this.registerLocationType(ConfiguredLocations.documentstore, new DocumentStoreIOLocation(s3StorageService));
    }

    @SuppressWarnings("PMD.UnusedPrivateMethod")
    private void registerLocationType(ConfiguredLocations locationType, IIOLocation location) {
        locations.put(locationType, location);
    }

    public InputStream read(ConfiguredLocations locationType, String location) throws IOLocationException {
        log.info("Getting configured location type: {}", locationType);
        var loc = locations.get(locationType);
        log.info("Location type found, performing read: {}", location);
        return loc.read(location);
    }

    public void write(ConfiguredLocations locationType, String location, InputStream payloadStream)
            throws IOLocationException {
        log.info("Getting configured location type: {}", locationType);
        var loc = locations.get(locationType);
        log.info("Location type found, performing write: {}", location);
        loc.write(location, payloadStream);
    }

    public void delete(ConfiguredLocations locationType, String location) throws IOLocationException {
        log.info("Getting configured location type: {}", locationType);
        var loc = locations.get(locationType);
        log.info("Location type found, performing delete: {}", location);
        loc.delete(location);
    }
}
