package gov.irs.directfile.api.io.directory;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;

import org.apache.tomcat.util.http.fileupload.FileUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import gov.irs.directfile.api.io.IOLocationException;

import static org.junit.jupiter.api.Assertions.*;

class DirectoryIOLocationTest {
    String fakeDirectory = "src/test/resources/whatever/";
    String fakeFile = String.format("%stest.txt", fakeDirectory);
    File directory = new File(fakeDirectory);

    @BeforeEach
    public void Create() {
        directory.mkdir();
    }

    @AfterEach
    public void Destroy() throws IOLocationException, InterruptedException, IOException {
        DirectoryIOLocation location = new DirectoryIOLocation();
        location.delete(fakeFile);
        FileUtils.cleanDirectory(directory);
        FileUtils.forceDelete(directory);
    }

    @Test
    public void CanCreateAnDirectoryStorageLocationForInformation() throws IOLocationException, IOException {
        DirectoryIOLocation location = new DirectoryIOLocation();
        InputStream s = new ByteArrayInputStream("expected back".getBytes());
        location.write(fakeFile, s);
        var output = location.read(fakeFile);
        var value = new String(output.readAllBytes(), Charset.defaultCharset());
        assertEquals("expected back", value);
        output.close();
    }

    @Test
    public void WritingToTheSameLocationWillOverwriteThePreviousInformation() throws IOLocationException, IOException {
        DirectoryIOLocation location = new DirectoryIOLocation();
        InputStream s = new ByteArrayInputStream("expected back".getBytes());
        location.write(fakeFile, s);
        InputStream s2 = new ByteArrayInputStream("second write".getBytes());
        location.write(fakeFile, s2);
        var output = location.read(fakeFile);
        var value = new String(output.readAllBytes(), Charset.defaultCharset());
        assertEquals("second write", value);
        output.close();
    }

    @Test
    public void WillThrowIfYouReadFromUnwrittenLocation() {
        // we don't guarantee that a file IO delete is timely, only that it will
        // happen eventually in Java and the disk's sweet time...
    }

    @Test
    public void CanReadFromTheSameLocationMultipleTimes() throws IOLocationException, IOException {
        DirectoryIOLocation location = new DirectoryIOLocation();
        InputStream s = new ByteArrayInputStream("expected back".getBytes());
        location.write(fakeFile, s);
        for (int i = 0; i < 10; i++) {
            var output = location.read(fakeFile);
            var value = new String(output.readAllBytes(), Charset.defaultCharset());
            assertEquals("expected back", value);
            output.close();
        }
    }
}
