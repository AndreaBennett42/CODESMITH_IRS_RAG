#!/usr/bin/env sh

set -e


../scripts/build-fact-graph.sh
cd ../libs/
echo "cleaning, compiling, installing shared dependencies..."
./mvnw clean install
