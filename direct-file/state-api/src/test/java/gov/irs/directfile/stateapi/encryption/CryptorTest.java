package gov.irs.directfile.stateapi.encryption;

import java.io.FileInputStream;
import java.security.PublicKey;
import java.security.Security;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;

import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import gov.irs.directfile.stateapi.model.AesGcmEncryptionResult;

import static org.assertj.core.api.Assertions.assertThat;

// cover unit tests for Encryptor and Decryptor
public class CryptorTest {

    @BeforeAll
    static void init() {
        Security.addProvider(new BouncyCastleProvider());
    }

    @Test
    public void testGeneratePassword() throws Exception {
        byte[] password = Encryptor.generatePassword();
        assertThat(password.length == 32);
    }

    @Test
    public void testGenerateIV() throws Exception {
        byte[] iv = Encryptor.generateIV();
        assertThat(iv.length == 12);
    }

    @Test
    public void testRsaEncryptWithPublicKey() throws Exception {
        byte[] secret = "secret".getBytes();

        byte[] encrypted = Encryptor.rsaEncryptWithPublicKey(
                secret, getPublicKey("src/test/resources/certificates/fakestate.cer"));

        assertThat(encrypted).isNotNull();
    }

    @Test
    public void testRsaDecryptWithPrivateKey() throws Exception {
        byte[] secret = "secret".getBytes();

        byte[] encrypted = Encryptor.rsaEncryptWithPublicKey(
                secret, getPublicKey("src/test/resources/certificates/fakestate.cer"));

        byte[] decrypted =
                Decryptor.rsaDecryptWithPrivateKey(encrypted, "src/test/resources/certificates/fakestate.key");

        assertThat(decrypted).isEqualTo(secret);
    }

    @Test
    public void testAesEncryptAndAesDecrypt() throws Exception {
        String xmlData = "<?xml version='1.0' ?><taxreturn>blahblah</taxreturn>";
        byte[] key = Encryptor.generatePassword();
        byte[] iv = Encryptor.generateIV();
        AesGcmEncryptionResult result = Encryptor.aesGcmEncrypt(xmlData, key, iv);

        byte[] decryptedXml = Decryptor.aesGcmDecrypt(result.ciphertext(), key, iv, result.authenticationTag());

        assertThat(new String(decryptedXml)).isEqualTo(xmlData);
    }

    private PublicKey getPublicKey(String certFilePath) throws Exception {
        CertificateFactory certFactory = CertificateFactory.getInstance("X.509");
        X509Certificate cert = (X509Certificate) certFactory.generateCertificate(new FileInputStream(certFilePath));

        return cert.getPublicKey();
    }
}
