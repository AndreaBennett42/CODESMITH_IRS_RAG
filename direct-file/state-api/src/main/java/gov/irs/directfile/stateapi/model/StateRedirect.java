package gov.irs.directfile.stateapi.model;

import java.time.Instant;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Table(name = "state_redirect")
public class StateRedirect {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotNull private Long stateProfileId;

    @NotBlank
    private String redirectUrl;

    @NotNull private Instant createdAt;

    private Instant expiresAt;
}
