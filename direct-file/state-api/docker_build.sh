#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR"
echo "$SCRIPT_DIR"

api_build_image_tag="state-api"
api_build_container_name="state-api-builder-container"
api_jar_file_name="state-api-0.0.1-SNAPSHOT.jar"
jar_output_path_component="target"

# build jar
docker buildx build \
  --pull \
  --build-context factgraph-repo="../fact-graph-scala" \
  --build-context boms="../boms" \
  --build-context config="../config" \
  --build-context shared-libs="../libs" \
  --build-arg MAVEN_OPTS="$MAVEN_OPTS" \
  --tag "$api_build_image_tag" \
  --file Dockerfile-local \
  "$@" \
  "$SCRIPT_DIR"

# extract jar to `./target`
mkdir -p "$SCRIPT_DIR/$jar_output_path_component"
docker container rm --force "$api_build_container_name" &>/dev/null
docker container create --name "$api_build_container_name" "$api_build_image_tag"
docker cp "$api_build_container_name":/deployments/state-api.jar "$SCRIPT_DIR/$jar_output_path_component/$api_jar_file_name"
docker container rm --force "$api_build_container_name"

printf "\njarfile: %s\n" "$SCRIPT_DIR/$jar_output_path_component/$api_jar_file_name"
