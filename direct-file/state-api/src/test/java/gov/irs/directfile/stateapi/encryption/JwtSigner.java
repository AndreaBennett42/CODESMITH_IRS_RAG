package gov.irs.directfile.stateapi.encryption;

import java.io.FileReader;
import java.io.IOException;
import java.security.PrivateKey;
import java.security.Signature;
import java.util.Base64;

import org.bouncycastle.asn1.pkcs.PrivateKeyInfo;
import org.bouncycastle.openssl.jcajce.JcaPEMKeyConverter;
import org.bouncycastle.util.io.pem.PemObject;
import org.bouncycastle.util.io.pem.PemReader;

/**
 * This class is not used in prod, only for unit test
 */
public class JwtSigner {

    public static String createJwt(String header, String payload, String privateKeyFilePath) throws Exception {
        PrivateKey privateKey = loadPrivateKeyFromFile(privateKeyFilePath);

        // Encode header and payload to Base64Url
        // choose java.util.Base64's url encoder instead of bouncy castle's UrlBase64,
        // UrlBase64 puts an extra dot at the end sometimes.
        String base64UrlHeader = Base64.getUrlEncoder().encodeToString(header.getBytes());
        String base64UrlPayload = Base64.getUrlEncoder().encodeToString(payload.getBytes());

        // Create the signing input
        String signingInput = base64UrlHeader + "." + base64UrlPayload;

        // Sign the JWT
        Signature signature = Signature.getInstance("SHA256withRSA", "BC");
        signature.initSign(privateKey);
        signature.update(signingInput.getBytes("UTF-8"));
        byte[] signedBytes = signature.sign();
        String base64UrlSignature = Base64.getUrlEncoder().encodeToString(signedBytes);

        // Combine header, payload, and signature to form the JWT
        return base64UrlHeader + "." + base64UrlPayload + "." + base64UrlSignature;
    }

    private static PrivateKey loadPrivateKeyFromFile(String privateKeyFilePath) throws IOException {
        PemReader pemReader = new PemReader(new FileReader(privateKeyFilePath));
        PemObject pemObject = pemReader.readPemObject();
        PrivateKeyInfo privateKeyInfo = PrivateKeyInfo.getInstance(pemObject.getContent());
        pemReader.close();

        return new JcaPEMKeyConverter().setProvider("BC").getPrivateKey(privateKeyInfo);
    }
}
