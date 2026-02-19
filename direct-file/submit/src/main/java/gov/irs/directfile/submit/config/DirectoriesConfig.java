package gov.irs.directfile.submit.config;

import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public class DirectoriesConfig {
    private String Input;
    private String ToProcess;
    private String Processed;
    private String ToBatch;
    private String Batched;
    private String Submitted;
}
