#!/usr/bin/env bash
## This may have hot reload bugs following npm workspaces. Talk to Brett or AMark if that affects your dev environment.
set -e

# Run development server in a container

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR"

docker buildx build \
  --pull \
  --file Dockerfile-development \
  --tag df-client:dev \
  .

docker run \
  --rm \
  --volume ./packages:/build/packages \
  --volume ./tsconfig.base.json:/build/tsconfig.base.json \
  --volume ./eslint-plugin-df-rules/tsconfig.json:/build/eslint-plugin-df-rules/tsconfig.json \
  --volume ./eslint-plugin-df-rules/src:/build/eslint-plugin-df-rules/src \
  --volume ./js-factgraph-scala/src:/build/js-factgraph-scala/src \
  --volume ./df-client-app/index.html:/build/df-client-app/index.html \
  --volume ./df-client-app/.env.development:/build/df-client-app/.env.development \
  --volume ./df-client-app/nodemon.json:/build/df-client-app/nodemon.json \
  --volume ./df-client-app/tsconfig.json:/build/df-client-app/tsconfig.json \
  --volume ./df-client-app/vite.config.ts:/build/df-client-app/vite.config.ts \
  --volume ./df-client-app/vite.config.allscreens.ts:/build/df-client-app/vite.config.allscreens.ts \
  --volume ./df-client-app/public:/build/df-client-app/public \
  --volume ./df-client-app/src:/build/df-client-app/src \
  --volume "$SCRIPT_DIR/../backend/src/main/resources/dataimportservice/mocks":/build/df-client-app/src/redux/slices/data-import/mocks \
  --volume ./df-static-site/src/locales:/build/df-static-site/src/locales \
  --volume "$SCRIPT_DIR/../backend/src/main/resources/tax":/build/df-client-app/src/fact-dictionary/generate-src/xml-src \
  --volume "$SCRIPT_DIR/../backend/src/main/resources/factgraphservice"/xmlFactPaths:/build/df-client-app/src/fact-dictionary/generate-src/xmlFactPaths \
  --volume "$SCRIPT_DIR/../backend/src/test/resources/scenarios":/build/df-client-app/src/test/scenarioTests/jsonScenarios \
  --env "DF_API_PORT=${DF_API_PORT:-8080}" \
  --env "DF_API_PUBLIC_PATH=${DF_API_PUBLIC_PATH:-/df/file/api/}" \
  --env "DF_CLIENT_PUBLIC_PATH=${DF_CLIENT_PUBLIC_PATH:-/df/file}" \
  --publish "${DF_LISTEN_ADDRESS:-127.0.0.1}:${DF_FE_PORT:-3000}:3000" \
  --name df-client-development-server \
  --interactive \
  --tty \
  "$@" \
  df-client:dev
