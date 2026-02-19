package gov.irs.directfile.submit.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@AllArgsConstructor
@Getter
@ConfigurationProperties(prefix = "submit")
public class Config {
    private String Environment;
    private KeystoreConfig Keystore;
    private IntervalsConfig Intervals;
    private DirectoriesConfig Directories;
    private DocumentStoreConfig DocumentStore;
    private MessageQueueConfig MessageQueue;
    private String Etin;

    @Setter
    private String Asid;

    private String Efin;
    private String Toolkit;
    private boolean Prod;
    private boolean RunnerDisabledForTesting;
    private boolean SubmitActionEnabled;
    private String VendorControlNumber;
    private String softwareId;
    private String softwareVersionNum;
    private String applicationId;
}
