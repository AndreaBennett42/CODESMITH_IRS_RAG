package gov.irs.directfile.emailservice.config;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@AllArgsConstructor
@Getter
@ConfigurationProperties(prefix = "email-service")
@Validated
public class EmailServiceConfigurationProperties {
    /**
     * Value that will be used for {@link Builder software.amazon.awssdk.services.sqs.model.SendMessageRequest.Builder.delaySeconds(Integer delaySeconds)}.
     *
     * It will default to 0 if null is passed to the {@code @AllArgsConstructor} generated constructor
     */
    @Min(0)
    @Max(900)
    private final int resendDelaySeconds;

    @NotBlank
    private final String environment;

    @NotNull private final Sender sender;

    @NotNull private final MessageQueue messageQueue;

    @AllArgsConstructor
    @Getter
    public static class Sender {
        @NotBlank
        private final String from;
    }

    @AllArgsConstructor
    @Getter
    public static class MessageQueue {
        @NotBlank
        private final String sendEmailQueue;

        @NotBlank
        private final String dlqSendEmailQueue;

        @NotBlank
        private final String endpoint;

        @NotBlank
        private final String region;

        @NotBlank
        private final String accessKey;

        @NotBlank
        private final String secretKey;
    }
}
