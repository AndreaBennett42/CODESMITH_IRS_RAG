package gov.irs.directfile.api.stateapi.domain;

import java.util.List;
import java.util.Map;

import gov.irs.directfile.models.StateOrProvince;

public record StateProfile(
        StateOrProvince stateCode,
        String taxSystemName,
        String landingUrl,
        String defaultRedirectUrl,
        String departmentOfRevenueUrl,
        String filingRequirementsUrl,
        String transferCancelUrl,
        String waitingForAcceptanceCancelUrl,
        List<String> redirectUrls,
        /*
         A map with keys that correspond to DF-understandable language codes to the corresponding codes
         used by the state. Maps back to state-api's StateLanguage entity e.g:

         {
             en: `en`
             es: `es `
         }

         OR

         {
             en: `english`
             es: `spanish`
         }
        */
        Map<String, String> languages,
        String customFilingDeadline,
        Boolean acceptedOnly) {}
