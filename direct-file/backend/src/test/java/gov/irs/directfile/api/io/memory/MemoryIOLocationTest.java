package gov.irs.directfile.api.io.memory;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;

import org.junit.jupiter.api.Test;

import gov.irs.directfile.api.io.IOLocationException;

import static org.junit.jupiter.api.Assertions.*;

class MemoryIOLocationTest {
    @Test
    public void CanCreateAnInMemoryStorageLocationForInformation() throws IOLocationException, IOException {
        MemoryIOLocation location = new MemoryIOLocation();
        InputStream s = new ByteArrayInputStream("expected back".getBytes());
        location.write("test", s);
        var output = location.read("test");
        var value = new String(output.readAllBytes(), Charset.defaultCharset());
        assertEquals("expected back", value);
    }

    @Test
    public void WritingToTheSameLocationWillOverwriteThePreviousInformation() throws IOLocationException, IOException {
        MemoryIOLocation location = new MemoryIOLocation();
        InputStream s = new ByteArrayInputStream("expected back".getBytes());
        location.write("test", s);
        InputStream s2 = new ByteArrayInputStream("second write".getBytes());
        location.write("test", s2);

        var output = location.read("test");
        var value = new String(output.readAllBytes(), Charset.defaultCharset());
        assertEquals("second write", value);
    }

    @Test
    public void WillThrowIfYouReadFromUnwrittenLocation() throws IOLocationException, IOException {
        assertThrows(IOLocationException.class, () -> {
            MemoryIOLocation location = new MemoryIOLocation();
            location.read("test");
        });
    }

    @Test
    public void CanReadFromTheSameLocationMultipleTimes() throws IOLocationException, IOException {
        MemoryIOLocation location = new MemoryIOLocation();
        InputStream s = new ByteArrayInputStream("expected back".getBytes());
        location.write("test", s);
        for (int i = 0; i < 10; i++) {
            var output = location.read("test");
            var value = new String(output.readAllBytes(), Charset.defaultCharset());
            assertEquals("expected back", value);
        }
    }
}
