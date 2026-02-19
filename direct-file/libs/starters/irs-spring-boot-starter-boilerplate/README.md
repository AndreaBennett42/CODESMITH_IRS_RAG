# IRS Spring Boot Starter Boilerplate

## Tips

* Use `@Validated` on `@ConfigurationProperties` to validate properties defined in `application[-*].yaml`

* Add `spring-boot-starter-web` and `spring-boot-starter-actuator` as dependency to access the following endpoints for debugging your custom starter
    * `/env` &mdash; Environment Variables
    * `/configprops` &mdash; loaded `@ConfigurationProperties`
    * `/beans` &mdash; to inspect your `@AutoConfiguration` beans 

## References

### Core Features
* [2. Externalized Configuration](https://docs.spring.io/spring-boot/docs/current/reference/html/features.html#features.external-config)
    * [2.5. Working With YAML](https://docs.spring.io/spring-boot/docs/current/reference/html/features.html#features.external-config.yaml)
    * [2.8. Type-safe Configuration Properties](https://docs.spring.io/spring-boot/docs/current/reference/html/features.html#features.external-config.typesafe-configuration-properties)
* [12. Creating Your Own Auto-configuration](https://docs.spring.io/spring-boot/docs/current/reference/html/features.html#features.developing-auto-configuration)
    * [12.5. Creating Your Own Starter](https://docs.spring.io/spring-boot/docs/current/reference/html/features.html#features.developing-auto-configuration.custom-starter)

### Spring Boot Maven Plugin Documentation
* [5. Packaging Executable Archives](https://docs.spring.io/spring-boot/docs/current/maven-plugin/reference/htmlsingle/#packaging)
    * [5.2.3. Parameter details &rarr; layout](https://docs.spring.io/spring-boot/docs/current/maven-plugin/reference/htmlsingle/#packaging.repackage-goal.parameter-details.layout)