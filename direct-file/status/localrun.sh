cd target
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=8001 -jar mef-status-0.0.1-SNAPSHOT.jar 
