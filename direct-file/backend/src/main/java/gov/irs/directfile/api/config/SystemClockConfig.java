package gov.irs.directfile.api.config;

import java.time.*;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@Slf4j
public class SystemClockConfig {
    @Bean
    public Clock systemClock() {
        return Clock.systemUTC();
    }
}
