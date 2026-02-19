#!/usr/bin/env sh
cd ../backend
mvn test -Dtest=FactGraphServiceScenarioTest
cd ../scripts