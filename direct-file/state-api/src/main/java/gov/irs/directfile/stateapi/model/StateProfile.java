package gov.irs.directfile.stateapi.model;

import java.time.OffsetDateTime;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Table(name = "state_profile")
public class StateProfile {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank
    private String accountId;

    @NotBlank
    private String stateCode;

    @NotBlank
    private String taxSystemName;

    @NotBlank
    private String landingUrl;

    private String defaultRedirectUrl;

    private String departmentOfRevenueUrl;

    private String filingRequirementsUrl;

    private String transferCancelUrl;

    private String waitingForAcceptanceCancelUrl;

    private String certLocation;

    @NotNull private Boolean acceptedOnly;

    private OffsetDateTime certExpirationDate;

    private String customFilingDeadline;

    @NotNull private Boolean archived;
}
