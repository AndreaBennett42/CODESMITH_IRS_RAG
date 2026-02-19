package gov.irs.directfile.stateapi.model;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Table(name = "state_language")
public class StateLanguage {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotNull private Long stateProfileId;

    /**
     * The language code that DF understands, e.g. `es` or `en`
     */
    @NotBlank
    private String dfLanguageCode;

    /**
     * The corresponding language code that the state uses, e.g. `en`, `eng`, `english`, etc
     */
    @NotBlank
    private String stateLanguageCode;
}
