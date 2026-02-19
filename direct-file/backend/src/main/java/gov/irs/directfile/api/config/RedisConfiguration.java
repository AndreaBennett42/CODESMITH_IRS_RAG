package gov.irs.directfile.api.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.JdkSerializationRedisSerializer;
import org.springframework.data.redis.serializer.StringRedisSerializer;

@Configuration
@Slf4j
public class RedisConfiguration {

    public static final String FEATURE_FLAG_CACHE_NAME = "feature-flags";
    public static final String STATUS_CACHE_NAME = "status";
    public static final String USERS_CACHE_NAME = "users";
    public static final String DATA_IMPORT_GATING_CACHE_NAME = "data-import-gating";

    @Bean
    RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory connectionFactory) {
        log.info("RedisConnectionFactory: {}", connectionFactory);
        RedisTemplate<String, Object> redisTemplate = new RedisTemplate<>();
        redisTemplate.setConnectionFactory(connectionFactory);
        redisTemplate.setKeySerializer(new StringRedisSerializer());
        // Note: may need to change the JdkSerializationRedisSerializer when implementing encryption
        redisTemplate.setValueSerializer(new JdkSerializationRedisSerializer());
        return redisTemplate;
    }
}
