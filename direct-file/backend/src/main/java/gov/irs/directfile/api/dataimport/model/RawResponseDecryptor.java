package gov.irs.directfile.api.dataimport.model;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import gov.irs.directfile.models.encryption.DataEncryptDecrypt;
import gov.irs.directfile.models.encryption.GenericStringEncryptor;

@Component
@Slf4j
public class RawResponseDecryptor {
    private final GenericStringEncryptor genericStringEncryptor;
    private final ObjectMapper objectMapper;

    public RawResponseDecryptor(DataEncryptDecrypt dataEncryptDecrypt, ObjectMapper objectMapper) {
        this.genericStringEncryptor = new GenericStringEncryptor(dataEncryptDecrypt);
        this.objectMapper = objectMapper;
    }

    public JsonNode decryptRawResponse(PopulatedData populatedData) {
        try {
            String decrypted = genericStringEncryptor.convertToEntityAttribute(populatedData.getRawDataCipherText());

            return objectMapper.readTree(decrypted);
        } catch (Exception e) {
            log.error(
                    "Failed to decrypt / parse data column in populated_data. Exception: {}. Error: {}",
                    e.getClass().getName(),
                    e.getMessage());
        }
        return null;
    }
}
