package gov.irs.directfile.stateapi.model;

public record AesGcmEncryptionResult(byte[] ciphertext, byte[] authenticationTag) {}
