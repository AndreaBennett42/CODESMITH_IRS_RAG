package gov.irs.directfile.api.dataimport;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.core.io.support.ResourcePatternUtils;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import gov.irs.directfile.api.dataimport.model.WrappedPopulatedData;

@Slf4j
@Service
@Profile("mock")
public class MockDataImportService implements DataImportService {

    protected static final String PATH = "classpath:dataimportservice/mocks/*.json";
    private Map<String, WrappedPopulatedData> map = new HashMap<>();
    private Map<String, Long> TIMEOUT_INCREMENTER_MAP = new HashMap<>();
    long RETRY_TIMEOUT = 20000;

    public MockDataImportService(ObjectMapper mapper, ResourceLoader loader) {

        try {
            for (Resource resource :
                    ResourcePatternUtils.getResourcePatternResolver(loader).getResources(PATH)) {

                String fileName = resource.getFilename();
                if (fileName != null) {
                    try {
                        String file = fileName.replace(".json", "");
                        WrappedPopulatedData data = mapper.readValue(
                                resource.getContentAsString(StandardCharsets.UTF_8), WrappedPopulatedData.class);
                        map.put(file, data);
                        TIMEOUT_INCREMENTER_MAP.put(file, data.getData().getTimeSinceCreation());
                    } catch (IOException e) {
                        log.error(
                                "failed to read: {} {} {}", resource.getFilename(), e.getMessage(), e.getStackTrace());
                    }
                }
            }
        } catch (IOException e) {
            log.error("failed to load mock data {} {}", e.getMessage(), e.getStackTrace());
        }
    }

    @Override
    public void sendPreFetchRequest(UUID taxReturnId, UUID userId, UUID externalId, String tin, int taxYear) {
        log.info(
                "Mock prefetch called for Tax Return: {}, User ID: {}, External ID: {}",
                taxReturnId,
                userId,
                externalId.toString());
    }

    @Override
    public WrappedPopulatedData getPopulatedData(UUID taxReturnId, UUID userId, Date taxReturnCreatedAt) {
        log.warn("Unexpected call for Tax Return: {}; User ID: {}", taxReturnId, userId);
        return WrappedPopulatedData.from(new ArrayList<>(), taxReturnCreatedAt);
    }

    public WrappedPopulatedData getPopulatedData(String key, String dateOfBirth) {
        log.info("Mock get data called for: {}", key);
        WrappedPopulatedData data =
                map.containsKey(key) ? map.get(key) : WrappedPopulatedData.from(new ArrayList<>(), new Date());
        if (StringUtils.hasText(dateOfBirth)) {
            for (JsonNode json : data.getData().getAboutYouBasic().getPayload()) {
                if (json.hasNonNull("dateOfBirth")) {
                    ((ObjectNode) json).put("dateOfBirth", dateOfBirth);
                }
            }
        }
        long nextTimeSinceCreation = TIMEOUT_INCREMENTER_MAP.get(key) + 1000;
        if (nextTimeSinceCreation < RETRY_TIMEOUT) {
            TIMEOUT_INCREMENTER_MAP.put(key, nextTimeSinceCreation);
        } else {
            TIMEOUT_INCREMENTER_MAP.put(key, 1000L);
        }
        return new WrappedPopulatedData(new WrappedPopulatedData.Data(
                data.getData().getAboutYouBasic(),
                data.getData().getIpPin(),
                data.getData().getW2s(),
                data.getData().getF1099Ints(),
                data.getData().getF1095a(),
                nextTimeSinceCreation));
    }
}
