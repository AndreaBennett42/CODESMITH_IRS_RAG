package gov.irs.directfile.stateapi.dto;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import gov.irs.directfile.stateapi.model.StateLanguage;
import gov.irs.directfile.stateapi.model.StateProfile;
import gov.irs.directfile.stateapi.model.StateRedirect;

public record StateProfileDTO(
        String stateCode,
        String taxSystemName,
        String landingUrl,
        String defaultRedirectUrl,
        String departmentOfRevenueUrl,
        String filingRequirementsUrl,
        String transferCancelUrl,
        String waitingForAcceptanceCancelUrl,
        List<String> redirectUrls,
        Map<String, String> languages,
        Boolean acceptedOnly,
        String customFilingDeadline,
        Boolean archived) {

    public StateProfileDTO(StateProfile stateProfile) {
        this(stateProfile, new ArrayList<>(), new ArrayList<>());
    }

    public StateProfileDTO(
            StateProfile stateProfile, List<StateRedirect> stateRedirects, List<StateLanguage> stateLanguages) {
        this(
                stateProfile.getStateCode(),
                stateProfile.getTaxSystemName(),
                stateProfile.getLandingUrl(),
                stateProfile.getDefaultRedirectUrl(),
                stateProfile.getDepartmentOfRevenueUrl(),
                stateProfile.getFilingRequirementsUrl(),
                stateProfile.getTransferCancelUrl(),
                stateProfile.getWaitingForAcceptanceCancelUrl(),
                stateRedirects.stream().map(StateRedirect::getRedirectUrl).toList(),
                stateLanguages.stream()
                        .collect(Collectors.toMap(
                                StateLanguage::getDfLanguageCode, StateLanguage::getStateLanguageCode)),
                stateProfile.getAcceptedOnly(),
                stateProfile.getCustomFilingDeadline(),
                stateProfile.getArchived());
    }
}
