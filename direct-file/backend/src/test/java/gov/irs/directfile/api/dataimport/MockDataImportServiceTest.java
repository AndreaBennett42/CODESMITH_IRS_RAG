package gov.irs.directfile.api.dataimport;

import java.io.IOException;
import java.util.Date;
import java.util.UUID;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.core.io.support.ResourcePatternUtils;
import org.springframework.test.context.junit.jupiter.SpringJUnitConfig;

import gov.irs.directfile.api.dataimport.model.WrappedPopulatedData;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

@Slf4j
@SpringJUnitConfig
class MockDataImportServiceTest {

    private final ObjectMapper mapper = new ObjectMapper();
    private final ResourceLoader loader = new DefaultResourceLoader();
    private DataImportService dataImportService;

    @BeforeEach
    void setUp() {
        dataImportService = new MockDataImportService(mapper, loader);
    }

    @Test
    void mockDataImportService_getPopulatedDataDoesNotThrow() {
        assertDoesNotThrow(() -> dataImportService.getPopulatedData(UUID.randomUUID(), UUID.randomUUID(), new Date()));
        assertDoesNotThrow(() -> dataImportService.getPopulatedData(null, null, new Date()));
    }

    @Test
    void mockDataJsonIsValid() throws IOException {
        int count = 0;

        for (Resource resource :
                ResourcePatternUtils.getResourcePatternResolver(loader).getResources(MockDataImportService.PATH)) {
            assertDoesNotThrow(() -> mapper.readValue(resource.getFile(), WrappedPopulatedData.class));
            count++;
        }

        assertThat(count).isPositive();
    }
}
