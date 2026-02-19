package gov.irs.directfile.api.dataimport.model;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.persistence.PostLoad;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import gov.irs.directfile.models.encryption.DataEncryptDecrypt;
import gov.irs.directfile.models.encryption.GenericStringEncryptor;

@Component
@Slf4j
public class PopulatedDataEntityListener {
    private GenericStringEncryptor genericStringEncryptor;
    private ObjectMapper objectMapper;

    @Autowired
    public void configure(DataEncryptDecrypt dataEncryptDecrypt, ObjectMapper objectMapper) {
        genericStringEncryptor = new GenericStringEncryptor(dataEncryptDecrypt);
        this.objectMapper = objectMapper;
    }

    @PostLoad
    public <T extends PopulatedData> void decryptColumn(T populatedData) {
        try {
            String decrypted = genericStringEncryptor.convertToEntityAttribute(populatedData.getDataCipherText());

            JsonNode jsonNode;
            jsonNode = objectMapper.readTree(decrypted);
            populatedData.setData(jsonNode);
        } catch (Exception e) {
            log.error(
                    "Failed to decrypt / parse data column in populated_data. Exception: {}. Error: {}",
                    e.getClass().getName(),
                    e.getMessage());
        }
    }
}
