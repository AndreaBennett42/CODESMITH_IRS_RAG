package gov.irs.directfile.api.taxreturn;

import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.UUID;

import com.amazonaws.encryptionsdk.AwsCrypto;
import com.amazonaws.encryptionsdk.CryptoMaterialsManager;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class EncryptionCacheWarmingService {
    private static final byte[] WARMING_DATA = "data-for-cache-warming".getBytes(StandardCharsets.UTF_8);

    private final CryptoMaterialsManager cryptoMaterialsManager;
    private final AwsCrypto awsCrypto;

    public void warmCacheForUserExternalId(UUID userExternalId) {
        Map<String, String> context = Map.of("id", userExternalId.toString());

        // This will cause a call to kms.GenerateDataKey if it was necessary
        awsCrypto.encryptData(cryptoMaterialsManager, WARMING_DATA, context);
    }
}
