package gov.irs.boot.autoconfigure.boilerplate;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.beans.factory.config.BeanDefinition;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Role;

@AutoConfiguration
@EnableConfigurationProperties(BoilerplateConfigurationProperties.class)
public class BoilerplateAutoConfiguration {

    @Bean
    @Role(BeanDefinition.ROLE_SUPPORT)
    @ConditionalOnMissingBean(BoilerplateBean.class)
    BoilerplateBean boilerplateBean(BoilerplateConfigurationProperties configProps) {
        return new BoilerplateBean(configProps);
    }

    @AllArgsConstructor
    @Getter
    public static class BoilerplateBean {
        private final BoilerplateConfigurationProperties BoilerplateStarterConfigurationProperties;
    }
}
