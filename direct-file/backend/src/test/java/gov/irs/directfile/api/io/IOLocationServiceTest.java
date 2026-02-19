package gov.irs.directfile.api.io;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;

import org.junit.jupiter.api.Test;

import gov.irs.directfile.api.io.documentstore.S3StorageService;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class IOLocationServiceTest {
    S3StorageService s3StorageService;

    @Test
    public void CanWriteToAnArbitraryMemoryStream() throws IOLocationException, IOException {
        IOLocationService service = new IOLocationService(s3StorageService);
        InputStream s = new ByteArrayInputStream("expected back".getBytes());
        service.write(IOLocationService.ConfiguredLocations.memory, "test", s);
        var output = service.read(IOLocationService.ConfiguredLocations.memory, "test");
        var value = new String(output.readAllBytes(), Charset.defaultCharset());
        assertEquals("expected back", value);
    }

    @Test
    public void WillThrowIfYouAttemptToReadFromAnUncreatedStream() throws IOLocationException, IOException {
        assertThrows(IOLocationException.class, () -> {
            IOLocationService service = new IOLocationService(s3StorageService);
            service.read(IOLocationService.ConfiguredLocations.memory, "test");
        });
    }

    @Test
    public void CanGatherInformationWithTheSameKeyFromDifferentSources() throws IOLocationException, IOException {
        IOLocationService service = new IOLocationService(s3StorageService);
        InputStream ms = new ByteArrayInputStream("expected back from memory".getBytes());
        InputStream fs = new ByteArrayInputStream("expected back from file".getBytes());
        service.write(IOLocationService.ConfiguredLocations.directory, "test.txt", fs);
        service.write(IOLocationService.ConfiguredLocations.memory, "test.txt", ms);
        var outputMemory = service.read(IOLocationService.ConfiguredLocations.memory, "test.txt");
        var outputFile = service.read(IOLocationService.ConfiguredLocations.directory, "test.txt");
        var valueMemory = new String(outputMemory.readAllBytes(), Charset.defaultCharset());
        var valueFile = new String(outputFile.readAllBytes(), Charset.defaultCharset());
        assertEquals("expected back from memory", valueMemory);
        assertEquals("expected back from file", valueFile);
        // cleaning up the file one
        service.delete(IOLocationService.ConfiguredLocations.directory, "test.txt");
    }

    @Test
    public void CanReReadTheSameStreamMultipleTimes() throws IOLocationException, IOException {
        IOLocationService service = new IOLocationService(s3StorageService);
        InputStream s = new ByteArrayInputStream("expected back".getBytes());
        service.write(IOLocationService.ConfiguredLocations.memory, "test", s);
        for (int i = 0; i < 10; i++) {
            var output = service.read(IOLocationService.ConfiguredLocations.memory, "test");
            var value = new String(output.readAllBytes(), Charset.defaultCharset());
            assertEquals("expected back", value);
        }
    }
}
