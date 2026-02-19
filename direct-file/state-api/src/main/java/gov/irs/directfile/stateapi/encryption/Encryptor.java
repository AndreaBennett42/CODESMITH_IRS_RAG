package gov.irs.directfile.stateapi.encryption;

import java.nio.charset.StandardCharsets;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.security.SecureRandom;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;

import org.bouncycastle.crypto.InvalidCipherTextException;
import org.bouncycastle.crypto.engines.AESEngine;
import org.bouncycastle.crypto.modes.GCMBlockCipher;
import org.bouncycastle.crypto.modes.GCMModeCipher;
import org.bouncycastle.crypto.params.AEADParameters;
import org.bouncycastle.crypto.params.KeyParameter;

import gov.irs.directfile.stateapi.model.AesGcmEncryptionResult;

@SuppressWarnings("PMD.SignatureDeclareThrowsException")
public class Encryptor {
    private static final SecureRandom SECURE_RANDOM = new SecureRandom();
    static int AES256_GCM_SECRET_LENGTH = 32;
    static int AES256_GCM_IV_LENGTH = 12;

    public static byte[] generatePassword() throws NoSuchAlgorithmException {
        // Create a KeyGenerator for AES (Advanced Encryption Standard) with a 256-bit
        // key size
        KeyGenerator keyGenerator = KeyGenerator.getInstance("AES");
        keyGenerator.init(AES256_GCM_SECRET_LENGTH * 8); // 256 bits = 32 bytes

        // Generate the secret key
        SecretKey secretKey = keyGenerator.generateKey();

        // Get the encoded bytes of the secret key
        return secretKey.getEncoded();
    }

    public static byte[] generateIV() {
        // AES-256 GCM requires 12 bytes initialization vector
        byte[] iv = new byte[AES256_GCM_IV_LENGTH];

        // Generate random bytes for the IV
        SECURE_RANDOM.nextBytes(iv);
        return iv;
    }

    public static byte[] rsaEncryptWithPublicKey(byte[] secret, PublicKey publicKey) throws Exception {
        Cipher cipher = Cipher.getInstance("RSA/ECB/OAEPWITHSHA-256ANDMGF1PADDING");
        cipher.init(Cipher.ENCRYPT_MODE, publicKey);

        return cipher.doFinal(secret);
    }

    public static AesGcmEncryptionResult aesGcmEncrypt(String plainText, byte[] aesKey, byte[] aesIV)
            throws InvalidCipherTextException {
        assert aesKey.length == AES256_GCM_SECRET_LENGTH
                : "AES 256 secret is not " + AES256_GCM_SECRET_LENGTH + " bytes";
        assert aesIV.length == AES256_GCM_IV_LENGTH : "AES 256 IV is not " + AES256_GCM_IV_LENGTH + " bytes";

        // Plain text to be encrypted
        byte[] plaintextBytes = plainText.getBytes(StandardCharsets.UTF_8);

        // Create AES-GCM block cipher
        GCMModeCipher gcmCipher = GCMBlockCipher.newInstance(AESEngine.newInstance());

        // Initialize with encryption mode, AES-256 key, and IV
        gcmCipher.init(true, new AEADParameters(new KeyParameter(aesKey), 128, aesIV));

        // Create a byte array for the ciphertext and associated data
        byte[] ciphertext = new byte[gcmCipher.getOutputSize(plaintextBytes.length)];

        // Encrypt the plaintext
        int len = gcmCipher.processBytes(plaintextBytes, 0, plaintextBytes.length, ciphertext, 0);
        gcmCipher.doFinal(ciphertext, len);

        // Retrieve the authentication tag
        byte[] authenticationTag = gcmCipher.getMac();

        return new AesGcmEncryptionResult(ciphertext, authenticationTag);
    }
}
