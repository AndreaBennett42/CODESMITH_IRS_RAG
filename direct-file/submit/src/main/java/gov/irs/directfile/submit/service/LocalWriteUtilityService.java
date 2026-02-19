package gov.irs.directfile.submit.service;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import javax.xml.datatype.DatatypeFactory;
import javax.xml.datatype.XMLGregorianCalendar;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;

@SuppressFBWarnings(
        value = {"DM_DEFAULT_ENCODING", "NM_METHOD_NAMING_CONVENTION"},
        justification = "Initial SpotBugs Setup")
@SuppressWarnings("PMD.CloseResource")
public class LocalWriteUtilityService {
    public static Path writeXmlToDisk(String xmlString, String submissionId, String outputDir, String fileName) {
        try {
            Path directory = Path.of(outputDir, submissionId);
            if (!Files.exists(directory)) {
                Files.createDirectory(directory);
            }
            Path returnPath = Path.of(outputDir, submissionId, fileName + ".xml");

            BufferedWriter writer = new BufferedWriter(new FileWriter(returnPath.toFile()));
            writer.write(xmlString);
            writer.close();
            return returnPath;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static OffsetDateTime testNow;

    private static OffsetDateTime getNow() {
        return testNow == null ? OffsetDateTime.now() : testNow;
    }

    public static void setUpTestNow(OffsetDateTime testNowLocal) {
        testNow = testNowLocal;
    }

    public static void tearDownTestNow() {
        testNow = null;
    }

    public static XMLGregorianCalendar Today() {
        OffsetDateTime date = getNow();
        // NOTE: ISO_OFFSET_DATE_TIME includes fractional seconds.
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("uuuu-MM-dd'T'HH:mm:ssXXX");
        var lex = date.format(formatter);
        return DatatypeFactory.newDefaultInstance().newXMLGregorianCalendar(lex);
    }

    public static XMLGregorianCalendar CreateGregorianDateFromString(String value) {
        return DatatypeFactory.newDefaultInstance().newXMLGregorianCalendar(value);
    }
}
