package gov.irs.directfile.models.encryption;

import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import lombok.SneakyThrows;

import gov.irs.directfile.models.FactTypeWithItem;

public class FactsEncryptor {
    private final DataEncryptDecrypt dataEncryptDecrypt;
    private final ObjectMapper mapper;

    public FactsEncryptor(DataEncryptDecrypt dataEncryptDecrypt) {
        this.dataEncryptDecrypt = dataEncryptDecrypt;
        mapper = new ObjectMapper();
        mapper.registerModule(new JavaTimeModule()); // Java 8 time not registered by default
    }

    @SneakyThrows
    public String convertToDatabaseColumn(
            Map<String, FactTypeWithItem> attribute, Map<String, String> encryptionContext) {
        if (attribute == null) {
            return null;
        }
        if (attribute.isEmpty()) {
            return "";
        }

        byte[] bytes = mapper.writeValueAsBytes(attribute);
        byte[] ciphertext = dataEncryptDecrypt.encrypt(bytes, encryptionContext);
        return Base64.getEncoder().encodeToString(ciphertext);
    }

    @SneakyThrows
    public Map<String, FactTypeWithItem> convertToEntityAttribute(String dbData) {
        if (dbData == null || dbData.isEmpty()) {
            return new HashMap<>();
        }

        byte[] ciphertext = Base64.getDecoder().decode(dbData);
        byte[] bytes = dataEncryptDecrypt.decrypt(ciphertext);
        return mapper.readValue(bytes, new TypeReference<>() {});
    }
}
