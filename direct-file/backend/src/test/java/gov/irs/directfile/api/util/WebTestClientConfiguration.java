package gov.irs.directfile.api.util;

import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.test.web.reactive.server.WebTestClientConfigurer;
import reactor.netty.http.client.HttpClient;

@TestConfiguration
public class WebTestClientConfiguration {
    @Bean
    public WebTestClientConfigurer webTestClientConfigurer() {
        return (webClientBuilder, webHttpHandlerBuilder, clientHttpConnector) -> {
            // "WebClientRequestException: Connection prematurely closed BEFORE response" can occur in some
            // tests that use WebTestClient.  Using ReactorClientHttpConnector avoids this.
            // https://stackoverflow.com/a/65846482
            webClientBuilder.clientConnector(new ReactorClientHttpConnector(HttpClient.newConnection()));
        };
    }
}
