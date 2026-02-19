package gov.irs.directfile.api.io.memory;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import lombok.extern.slf4j.Slf4j;

import gov.irs.directfile.api.io.IIOLocation;
import gov.irs.directfile.api.io.IOLocationException;

@Slf4j
@SuppressWarnings("PMD.CloseResource")
public class MemoryIOLocation implements IIOLocation {
    // think of this location as a cache.
    // don't use it as a cache, because the stream copy isn't performant.
    // it is still better than getting files off of disk.
    // Locations are meant for heavy IO operations
    Map<String, InputStream> writtenData;

    public MemoryIOLocation() {
        writtenData = new HashMap<>();
    }

    @Override
    public InputStream read(String location) throws IOLocationException {
        if (writtenData.containsKey(location)) {
            var stream = writtenData.get(location);
            // we might hit a case where multiple threads try to read from the same stream
            // at the same time.  We don't want a partial read.  There are better ways of solving
            // this, but this is a quick way.  If this performs anything like dotnet, this will show
            // itself as a bottleneck by increasing the background thread count beyond normal levels.
            // The read should be fast enough that it doesn't really matter, but I'd like to be certain.
            // P.S. I don't care about writing as much in this context, but if write becomes important
            // then you will need to sync that too.
            synchronized (stream) {
                try {
                    // We cannot hand out the original stream, as we don't know the stateCode
                    // they will return it in.  We want to have it fresh and ready for each call
                    // so we pass out a copy and reset ours for the next call
                    log.info("In memory information found at {}!  Copying bytes to a new buffer", location);
                    var bytes = stream.readAllBytes();
                    stream.reset();
                    return new ByteArrayInputStream(bytes);
                } catch (IOException e) {
                    throw new IOLocationException("In memory location couldn't copy bytes to allow reading", e);
                }
            }
        }
        throw new IOLocationException(
                String.format("Did not write data to in memory location %s before reading", location));
    }

    @Override
    public void write(String location, InputStream payloadStream) throws IOLocationException {
        // we don't know the origin of the stream,
        // so we want to put it into memory, not just assume it was from there.
        byte[] bytes;
        try {
            log.info("Reading data from the input stream to write to in memory stream");
            bytes = payloadStream.readAllBytes();
        } catch (IOException e) {
            throw new IOLocationException("Could not write bytes to in memory stream", e);
        }
        var stream = new ByteArrayInputStream(bytes);
        log.info("Adding in memory byte stream at {}", location);
        writtenData.put(location, stream);
    }

    @Override
    public void delete(String location) throws IOLocationException {
        // my Java isn't great, but I assume that removing this reference
        // will allow the garbage collector to find and destroy this
        writtenData.remove(location);
    }
}
