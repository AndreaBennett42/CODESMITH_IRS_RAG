package gov.irs.directfile.stateapi.encryption;

import java.io.FileReader;
import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.PrivateKey;
import java.util.Arrays;
import javax.crypto.Cipher;
import javax.crypto.NoSuchPaddingException;

import org.bouncycastle.asn1.pkcs.PrivateKeyInfo;
import org.bouncycastle.crypto.InvalidCipherTextException;
import org.bouncycastle.crypto.engines.AESEngine;
import org.bouncycastle.crypto.modes.GCMBlockCipher;
import org.bouncycastle.crypto.modes.GCMModeCipher;
import org.bouncycastle.crypto.params.AEADParameters;
import org.bouncycastle.crypto.params.KeyParameter;
import org.bouncycastle.openssl.jcajce.JcaPEMKeyConverter;
import org.bouncycastle.util.io.pem.PemObject;
import org.bouncycastle.util.io.pem.PemReader;

/**
 * Decryptor class provides functions to decrypt data encrypted with public key
 * and AES 256 GCM.
 */
public class Decryptor {
    static int AES256_GCM_SECRET_LENGTH = 32;
    static int AES256_GCM_IV_LENGTH = 12;

    /**
     * Decrypt data with private key
     *
     * @param encryptedBytes     the data encrypted with public key
     * @param privateKeyFilePath the private key file location
     * @return decrypted data in byte array
     * @throws IOException
     * @throws NoSuchPaddingException
     * @throws NoSuchProviderException
     * @throws NoSuchAlgorithmException
     * @throws InvalidKeyException
     * @throws Exception, e.g.
     *         IOException,
     *         NoSuchAlgorithmException,
     *         NoSuchProviderException,
     *         NoSuchPaddingException,
     *         IllegalBlockSizeException,
     *         BadPaddingException,
     *         InvalidKeyException
     */
    public static byte[] rsaDecryptWithPrivateKey(byte[] encryptedBytes, String privateKeyFilePath) throws Exception {
        PrivateKey privateKey = loadPrivateKeyFromFile(privateKeyFilePath);

        Cipher cipher = Cipher.getInstance("RSA/ECB/OAEPWITHSHA-256ANDMGF1PADDING");
        cipher.init(Cipher.DECRYPT_MODE, privateKey);

        return cipher.doFinal(encryptedBytes);
    }

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

    private static PrivateKey loadPrivateKeyFromFile(String privateKeyFilePath) throws IOException {

        PemReader pemReader = new PemReader(new FileReader(privateKeyFilePath));
        PemObject pemObject = pemReader.readPemObject();
        PrivateKeyInfo privateKeyInfo = PrivateKeyInfo.getInstance(pemObject.getContent());
        pemReader.close();

        return new JcaPEMKeyConverter().setProvider("BC").getPrivateKey(privateKeyInfo);
    }
}
