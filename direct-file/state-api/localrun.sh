#cp -R ./app-config/* target/.
cd target
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=8000 -jar state-api-0.0.1-SNAPSHOT.jar 
