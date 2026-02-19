package gov.irs.directfile.api.dataimport.gating;

import java.lang.reflect.Field;
import java.time.ZonedDateTime;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.anyOf;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
public class DataImportGatingServiceTest {

    @Mock
    private DataImportGatingConfigService gatingConfigService;

    @Mock
    private DataImportGatingEmailAllowlistService emailAllowlistService;

    private DataImportGatingService dataImportGatingService;

    private DataImportGatingService spy;

    private ObjectMapper objectMapper = new ObjectMapper();

    @BeforeEach
    void setup() throws Exception {

        dataImportGatingService = new DataImportGatingService(gatingConfigService, emailAllowlistService);
        // Inject the value using reflection
        Field field = DataImportGatingService.class.getDeclaredField("dataImportEnabled");
        field.setAccessible(true);
        field.set(dataImportGatingService, true);
        // Spy on the service to override getCurrentTime
        spy = Mockito.spy(dataImportGatingService);

        objectMapper.findAndRegisterModules();
    }

    @Test
    void getBehavior_dataImportDisabled_thenReturnsBasic() throws Exception {

        // Inject the value using reflection
        Field field = DataImportGatingService.class.getDeclaredField("dataImportEnabled");
        field.setAccessible(true);
        field.set(dataImportGatingService, false);

        // when
        DataImportBehavior result = dataImportGatingService.getBehavior("john@example.com");

        // then
        assertThat(result, is(DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC));
    }

    @Test
    void getBehavior_allowListEnabledAndOnAllowList_thenReturnsBasicIpPinW2() throws Exception {

        when(emailAllowlistService.emailOnAllowlist(any())).thenReturn(true);

        // when
        DataImportBehavior result = spy.getBehavior("john@example.com");

        // then
        assertThat(result, is(DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2));
    }

    @Test
    void getBehavior_allowListEnabledButNotOnAllowList_inWindowing_thenReturnsPercentageBasedBehavior()
            throws Exception {
        // given
        String dataImportGatingConfigString =
                """
				{					
					"percentages": [
						{
							"behavior": "DATA_IMPORT_ABOUT_YOU_BASIC",
							"percentage": 30
						},
						{
							"behavior": "DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2",
							"percentage": 70
						}
					],
					"windowing": [
						{
							"start": "2025-01-01T00:00:00Z",
							"end": "2025-01-31T23:59:59Z"
						},
						{
							"start": "2025-02-01T00:00:00Z",
							"end": "2025-02-15T23:59:59Z"
						}	
					]				
				}
				""";

        DataImportGatingConfig config =
                objectMapper.readValue(dataImportGatingConfigString, DataImportGatingConfig.class);
        when(gatingConfigService.getGatingS3Config()).thenReturn(config);
        when(emailAllowlistService.emailOnAllowlist(any())).thenReturn(false);

        // Mock current time within the window
        ZonedDateTime mockNow = ZonedDateTime.parse("2025-01-15T10:00:00Z");
        Mockito.doReturn(mockNow).when(spy).getCurrentTime();

        // when
        DataImportBehavior result = spy.getBehavior("john@example.com");

        // then
        assertThat(
                result,
                anyOf(
                        is(DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC),
                        is(DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2)));
    }

    @Test
    void getBehavior_allowListEnabledButNotOnAllowList_notInWindowing_thenReturnsBasicIpPin() throws Exception {
        // given
        String dataImportGatingConfigString =
                """
				{					
					"percentages": [
						{
							"behavior": "DATA_IMPORT_ABOUT_YOU_BASIC",
							"percentage": 30
						},
						{
							"behavior": "DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2",
							"percentage": 70
						}
					],
					"windowing": [
						{
							"start": "2025-01-01T00:00:00Z",
							"end": "2025-01-31T23:59:59Z"
						},
						{
							"start": "2025-02-01T00:00:00Z",
							"end": "2025-02-15T23:59:59Z"
						}	
					]				
				}
				""";

        DataImportGatingConfig config =
                objectMapper.readValue(dataImportGatingConfigString, DataImportGatingConfig.class);
        when(gatingConfigService.getGatingS3Config()).thenReturn(config);
        when(emailAllowlistService.emailOnAllowlist(any())).thenReturn(false);

        // Mock current time within the window
        ZonedDateTime mockNow = ZonedDateTime.parse("2025-06-15T10:00:00Z");
        Mockito.doReturn(mockNow).when(spy).getCurrentTime();

        // when
        DataImportBehavior result = spy.getBehavior("john@example.com");

        // then
        assertThat(result, is(DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN));
    }

    @Test
    void getBehavior_notOnAllowlist_noWindowing_thenReturnsPercentageBasedBehavior() throws Exception {
        // given
        String dataImportGatingConfigString =
                """
				{					
					"percentages": [
						{
							"behavior": "DATA_IMPORT_ABOUT_YOU_BASIC",
							"percentage": 30
						},
						{
							"behavior": "DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2",
							"percentage": 70
						}
					],
					"windowing": [
					]				
				}
				""";

        DataImportGatingConfig config =
                objectMapper.readValue(dataImportGatingConfigString, DataImportGatingConfig.class);
        when(gatingConfigService.getGatingS3Config()).thenReturn(config);

        // when
        DataImportBehavior result = spy.getBehavior("john@example.com");

        // then
        assertThat(
                result,
                anyOf(
                        is(DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC),
                        is(DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2)));
    }

    @Test
    void getBehavior_notOnAllowlist_noS3Config_thenReturnsBasicPlusIpPin() throws Exception {
        when(emailAllowlistService.emailOnAllowlist(any())).thenReturn(false);
        when(gatingConfigService.getGatingS3Config()).thenReturn(null);
        // when
        DataImportBehavior result = spy.getBehavior("john@example.com");

        // then
        assertThat(result, is(DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN));
    }

    @Test
    void getBehavior_noPercentage_noWindowing_onAllowlist_thenReturnsBasicPlusIpPinPLusW2() throws Exception {
        when(emailAllowlistService.emailOnAllowlist(any())).thenReturn(true);

        // when
        DataImportBehavior result = spy.getBehavior("john@example.com");

        // then
        assertThat(result, is(DataImportBehavior.DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2));
    }
}
