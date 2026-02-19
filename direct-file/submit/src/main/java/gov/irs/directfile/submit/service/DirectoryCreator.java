package gov.irs.directfile.submit.service;

import java.beans.PropertyDescriptor;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import org.springframework.beans.BeanUtils;

import gov.irs.directfile.submit.config.Config;
import gov.irs.directfile.submit.config.DirectoriesConfig;

// on startup this verifies the directories and creates them if necessary
@SuppressFBWarnings(
        value = {"NM_METHOD_NAMING_CONVENTION"},
        justification = "Initial SpotBugs Setup")
public class DirectoryCreator {
    private final Config config;

    public DirectoryCreator(Config config) {
        this.config = config;
    }

    public void CreateDirectories() throws IOException, IllegalAccessException, InvocationTargetException {
        PropertyDescriptor[] directoriesConfigPropertyDescriptors =
                BeanUtils.getPropertyDescriptors(DirectoriesConfig.class);
        for (PropertyDescriptor pd : directoriesConfigPropertyDescriptors) {
            if (!String.class.equals(pd.getPropertyType())) {
                continue;
            }
            String dirString = (String) pd.getReadMethod().invoke(config.getDirectories());
            Path path = Path.of(dirString);
            Files.createDirectories(path);
        }
    }
}
