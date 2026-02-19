# IRS Spring Boot starter parent

## Maven properties that can be overridden

### Maven plugin configuration



| Property | Default&nbsp;Value | Notes |
|----|----|----|
|spotless-plugin.palantirVersion|`2.30.0`|
|spotless-plugin.file.xml-prefs|`${project.basedir}/../backend/xml.prefs`|Default value may not be correct depending on the projects base directory; assumes it's at the same level as `/direct-file/direct-file/backend/`|
|sonar.java.coveragePlugin|`jacoco`|
|sonar.coverage.jacoco.xmlReportPaths|`${project.basedir}/target/site/jacoco/jacoco.xml`|
|sonar.dynamicAnalysis|`reuseReports`|
|sonar.language|`java`|

### Version management

#### Dependencies

| Property | Default&nbsp;Value | Notes |
|----|----|----|
|aws-sdk.version|`2.20.131`|[AWS SDK for Java 2.x &rarr; Set up an Apache Maven project](https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/setup-project-maven.html#sdk-as-dependency)|
|aws-encryption-sdk.version|`2.4.1`|[AWS Encryption SDK for Java &rarr; Installation](https://docs.aws.amazon.com/encryption-sdk/latest/developer-guide/java.html#java-installation)|
|directfile.version|`0.0.1-SNAPSHOT`|
|spotbugs.version|`4.7.3`|
|spring-cloud.version|`2022.0.5`|[Spring Cloud &mdash; Supported Versions](https://github.com/spring-cloud/spring-cloud-release/wiki/Supported-Versions)|
|spring-cloud-aws.version|`3.0.4`|[Compatibility with Spring Project Versions](https://github.com/awspring/spring-cloud-aws?tab=readme-ov-file#compatibility-with-spring-project-versions)|


#### Maven Plugins
| Property | Default&nbsp;Value | Notes |
|----|----|----|
|cyclonedx-maven-plugin.version|`2.8.0`|
|jacoco-maven-plugin.version|`0.8.12`|
|maven-pmd-plugin.version|`3.22.2`|
|maven-site-plugin.version|`3.12.1`|
|spotbugs-maven-plugin.version|`4.8.5.0`|
|spotless-maven-plugin.version|`2.43.0`|