package gov.irs.directfile.stateapi.encryption;

import java.util.Arrays;

import org.bouncycastle.crypto.InvalidCipherTextException;
import org.bouncycastle.crypto.engines.AESEngine;
import org.bouncycastle.crypto.modes.GCMBlockCipher;
import org.bouncycastle.crypto.modes.GCMModeCipher;
import org.bouncycastle.crypto.params.AEADParameters;
import org.bouncycastle.crypto.params.KeyParameter;

/**
 * This class contains sample code for decrypting the tax return, which will be shared with the client to assist in the decryption process.
 */
@SuppressWarnings("PMD.ReturnEmptyCollectionRatherThanNull")
public class DecryptionDemoCode {
    static int AES256_GCM_SECRET_LENGTH = 32;
    static int AES256_GCM_IV_LENGTH = 12;

    /**
     * Decrypt data using AES 256 GCM.
     * @param ciphertext the data encrypted
     * @param encryptionKey the encryption secret
     * @param iv the initialization vector
     * @param authenticationTag the authentication tag
     * @return encrypted data in byte array or null if authentication tag is invalid
     * @throws InvalidCipherTextException
     */
    public static byte[] aesGcmDecrypt(byte[] ciphertext, byte[] encryptionKey, byte[] iv, byte[] authenticationTag)
            throws InvalidCipherTextException {
        assert encryptionKey.length == AES256_GCM_SECRET_LENGTH
                : "AES 256 secret is not " + AES256_GCM_SECRET_LENGTH + " bytes";
        assert iv.length == AES256_GCM_IV_LENGTH : "AES 256 IV is not " + AES256_GCM_IV_LENGTH + " bytes";

        // Create AES-GCM block cipher
        GCMModeCipher gcmCipher = GCMBlockCipher.newInstance(AESEngine.newInstance());

        // Initialize the cipher for decryption
        gcmCipher.init(false, new AEADParameters(new KeyParameter(encryptionKey), 128, iv));

        byte[] decrypted = new byte[gcmCipher.getOutputSize(ciphertext.length)];

        int len = gcmCipher.processBytes(ciphertext, 0, ciphertext.length, decrypted, 0);

        gcmCipher.doFinal(decrypted, len);

        // To verify the authentication tag, compare it to the original authentication tag
        if (Arrays.equals(authenticationTag, gcmCipher.getMac())) {
            return decrypted;
        } else {
            return null; // Authentication tag verification failed
        }
    }
}
