package gov.irs.directfile.api.util.base;

import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Import;

import gov.irs.directfile.api.util.SecurityTestConfiguration;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Import({SecurityTestConfiguration.class})
public abstract class BaseTest {}
