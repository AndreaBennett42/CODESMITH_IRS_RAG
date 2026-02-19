package gov.irs.directfile.stateapi.model;

public record EncryptData(
        String encodedSecret, String encodedIV, String encodedAndEncryptedData, String encodedAuthenticationTag) {}
