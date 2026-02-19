package gov.irs.directfile.models.encryption;

import java.util.Base64;
import java.util.Map;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.AllArgsConstructor;

@AllArgsConstructor
@SuppressFBWarnings(value = "DM_DEFAULT_ENCODING", justification = "Initial Spotbugs Setup")
public class GenericStringEncryptor {
    private final DataEncryptDecrypt dataEncryptDecrypt;

    public String convertToDatabaseColumn(String attribute, Map<String, String> encryptionContext) {
        if (attribute == null || attribute.isEmpty()) {
            return attribute;
        }
        byte[] ciphertext = dataEncryptDecrypt.encrypt(attribute.getBytes(), encryptionContext);
        return Base64.getEncoder().encodeToString(ciphertext);
    }

    public String convertToEntityAttribute(String dbData) {
        if (dbData == null || dbData.isEmpty()) {
            return dbData;
        }
        byte[] ciphertext = Base64.getDecoder().decode(dbData);
        byte[] decrypted = dataEncryptDecrypt.decrypt(ciphertext);
        return new String(decrypted);
    }
}
