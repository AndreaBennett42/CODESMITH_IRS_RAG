package gov.irs.directfile.stateapi.authorization;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.text.ParseException;
import java.time.Instant;
import java.util.*;

import com.amazonaws.encryptionsdk.exception.AwsCryptoException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nimbusds.jose.*;
import com.nimbusds.jose.crypto.MACSigner;
import com.nimbusds.jose.crypto.MACVerifier;
import com.nimbusds.jwt.JWTClaimsSet;
import com.nimbusds.jwt.SignedJWT;
import lombok.SneakyThrows;
import org.junit.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.test.StepVerifier;

import gov.irs.directfile.models.encryption.DataEncryptDecrypt;
import gov.irs.directfile.stateapi.model.AuthCodeRequest;

import static gov.irs.directfile.stateapi.authorization.AuthorizationTokenService.EXPORT_CLAIM_KEY;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class AuthorizationTokenServiceTest {
    private static final String TEST_SIGNING_KEY = "GTc+SlI7C7ECPHAhAvIWqn2yAvzAGMVj";
    private final ObjectMapper mapper = new ObjectMapper();

    public AuthorizationTokenService initializeService(DataEncryptDecrypt dataEncryptDecrypt) {
        return new AuthorizationTokenService(dataEncryptDecrypt, TEST_SIGNING_KEY, 60);
    }

    private AuthorizationTokenClaims setupClaims() {
        AuthCodeRequest authCodeRequest =
                new AuthCodeRequest(UUID.randomUUID(), "123-00-4567", 2023, "MA", "123456789AB");
        return mapper.convertValue(authCodeRequest, AuthorizationTokenClaims.class);
    }

    @SneakyThrows
    private byte[] setupSignedJwt(AuthorizationTokenClaims claims) {
        JWSSigner signer = new MACSigner(TEST_SIGNING_KEY);
        Instant issuedAt = Instant.now();
        JWTClaimsSet jwtClaimsSet = new JWTClaimsSet.Builder()
                .claim(EXPORT_CLAIM_KEY, claims)
                .issueTime(Date.from(issuedAt))
                .expirationTime(Date.from(issuedAt.plusSeconds(60)))
                .build();
        JWSObject jwsObject = new JWSObject(
                new JWSHeader.Builder(JWSAlgorithm.HS256).build(), new Payload(jwtClaimsSet.toJSONObject()));
        jwsObject.sign(signer);
        String signedJwt = jwsObject.serialize();
        return signedJwt.getBytes(StandardCharsets.UTF_8);
    }

    @Test
    public void givenTaxReturnSubmissionClaims_whenGeneratingAuthorizationToken_thenEncryptorIsCalled() {
        // given
        DataEncryptDecrypt dataEncryptDecrypt = mock(DataEncryptDecrypt.class);
        AuthorizationTokenService authorizationTokenService = initializeService(dataEncryptDecrypt);
        AuthorizationTokenClaims authorizationTokenClaims = setupClaims();

        // when
        // NOTE: We aren't stubbing the encryptor's return value, so a NullPointerException will be thrown.
        // That's ok because we're not testing what the encryptor returns,
        // we're just testing that the encryptor is invoked
        assertThrows(NullPointerException.class, () -> authorizationTokenService
                .generateAndEncrypt(authorizationTokenClaims)
                .block());

        // then
        verify(dataEncryptDecrypt).encrypt(any(), any());
    }

    @Test
    public void
            givenTaxReturnSubmissionClaims_whenGeneratingAuthorizationToken_thenEncryptorIsCalledWithProperContext() {
        // given
        DataEncryptDecrypt dataEncryptDecrypt = mock(DataEncryptDecrypt.class);
        AuthorizationTokenService authorizationTokenService = initializeService(dataEncryptDecrypt);
        AuthorizationTokenClaims authorizationTokenClaims = setupClaims();

        // when
        // NOTE: We aren't stubbing the encryptor's return value, so a NullPointerException will be thrown.
        // That's ok because we're not testing what the encryptor returns,
        // we're just testing that an encryption context is passed into the encryptor
        assertThrows(NullPointerException.class, () -> authorizationTokenService
                .generateAndEncrypt(authorizationTokenClaims)
                .block());
        ArgumentCaptor<Map<String, String>> encryptionContextCaptor = ArgumentCaptor.captor();

        // then
        verify(dataEncryptDecrypt).encrypt(any(), encryptionContextCaptor.capture());
        assertEquals(Map.of("system", "DIRECT-FILE", "type", "STATE-API"), encryptionContextCaptor.getValue());
    }

    @Test
    public void givenGeneratingAuthorizationToken_whenAnExceptionIsThrown_thenItPropagates() {
        // given
        DataEncryptDecrypt dataEncryptDecrypt = mock(DataEncryptDecrypt.class);
        AuthorizationTokenService authorizationTokenService = initializeService(dataEncryptDecrypt);

        // when
        when(dataEncryptDecrypt.encrypt(any(), any())).thenThrow(AwsCryptoException.class);
        AuthorizationTokenClaims claimsMap = setupClaims();

        // then
        StepVerifier.create(authorizationTokenService.generateAndEncrypt(claimsMap))
                .expectError(AwsCryptoException.class)
                .verify();
    }

    @Test
    public void givenTaxReturnSubmissionClaims_whenGeneratingAuthorizationToken_thenTokenIsSignedWithProperKey() {
        // given
        DataEncryptDecrypt dataEncryptDecrypt = mock(DataEncryptDecrypt.class);
        AuthorizationTokenService authorizationTokenService = initializeService(dataEncryptDecrypt);
        AuthorizationTokenClaims tokenClaims = setupClaims();
        byte[] signedJwtBytes = setupSignedJwt(tokenClaims);
        when(dataEncryptDecrypt.encrypt(any(), any())).thenReturn(signedJwtBytes);

        // when
        authorizationTokenService.generateAndEncrypt(tokenClaims).block();

        // then
        ArgumentCaptor<byte[]> signedTokenCaptor = ArgumentCaptor.captor();
        verify(dataEncryptDecrypt).encrypt(signedTokenCaptor.capture(), anyMap());
        byte[] signedTokenArgument = signedTokenCaptor.getValue();
        try {
            JWSVerifier signatureVerifier = new MACVerifier(TEST_SIGNING_KEY);
            SignedJWSParts signedJWSParts = mapper.readValue(signedTokenArgument, SignedJWSParts.class);
            String signedJWS = String.join(".", signedJWSParts.s1(), signedJWSParts.s2(), signedJWSParts.s3());
            SignedJWT signedJWT = SignedJWT.parse(signedJWS);
            assertTrue(signedJWT.verify(signatureVerifier));
        } catch (JOSEException | ParseException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void
            givenTaxReturnSubmissionClaims_whenGeneratingAuthorizationToken_thenTokenContainsEncryptedExportIdentifiers() {
        DataEncryptDecrypt dataEncryptDecrypt = mock(DataEncryptDecrypt.class);
        AuthorizationTokenService authorizationTokenService = initializeService(dataEncryptDecrypt);

        // given
        AuthorizationTokenClaims tokenClaims = setupClaims();
        byte[] signedJwtBytes = setupSignedJwt(tokenClaims);
        when(dataEncryptDecrypt.encrypt(any(), any())).thenReturn(signedJwtBytes);

        // when
        authorizationTokenService.generateAndEncrypt(tokenClaims).block();

        // then
        ArgumentCaptor<byte[]> signedTokenCaptor = ArgumentCaptor.captor();
        verify(dataEncryptDecrypt).encrypt(signedTokenCaptor.capture(), anyMap());

        byte[] signedTokenArgument = signedTokenCaptor.getValue();
        try {
            SignedJWSParts signedJWSParts = mapper.readValue(signedTokenArgument, SignedJWSParts.class);
            String signedJWS = String.join(".", signedJWSParts.s1(), signedJWSParts.s2(), signedJWSParts.s3());
            SignedJWT signedJWT = SignedJWT.parse(signedJWS);
            JWTClaimsSet jwtClaimsSet = signedJWT.getJWTClaimsSet();
            AuthorizationTokenClaims taxReturnExportClaim = mapper.convertValue(
                    jwtClaimsSet.toJSONObject().get("tax-return-export-metadata"), AuthorizationTokenClaims.class);
            assertEquals(tokenClaims, taxReturnExportClaim);
        } catch (ParseException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void givenTaxReturnSubmissionClaims_whenGeneratingAuthorizationToken_thenExpirationIsSet() {
        DataEncryptDecrypt dataEncryptDecrypt = mock(DataEncryptDecrypt.class);
        AuthorizationTokenService authorizationTokenService = initializeService(dataEncryptDecrypt);

        // given
        AuthorizationTokenClaims tokenClaims = setupClaims();
        byte[] signedJwtBytes = setupSignedJwt(tokenClaims);
        when(dataEncryptDecrypt.encrypt(any(), any())).thenReturn(signedJwtBytes);

        // when
        authorizationTokenService.generateAndEncrypt(tokenClaims).block();

        // then
        ArgumentCaptor<byte[]> signedTokenCaptor = ArgumentCaptor.captor();
        verify(dataEncryptDecrypt).encrypt(signedTokenCaptor.capture(), anyMap());

        byte[] signedTokenArgument = signedTokenCaptor.getValue();
        try {
            SignedJWSParts signedJWSParts = mapper.readValue(signedTokenArgument, SignedJWSParts.class);
            String signedJWS = String.join(".", signedJWSParts.s1(), signedJWSParts.s2(), signedJWSParts.s3());
            SignedJWT signedJWT = SignedJWT.parse(signedJWS);
            JWTClaimsSet jwtClaimsSet = signedJWT.getJWTClaimsSet();
            assertTrue(jwtClaimsSet.getExpirationTime().after(Date.from(Instant.now())));
            assertTrue(jwtClaimsSet
                    .getExpirationTime()
                    .before(Date.from(Instant.now().plusSeconds(60))));
        } catch (ParseException | IOException e) {
            throw new RuntimeException(e);
        }
    }
}
