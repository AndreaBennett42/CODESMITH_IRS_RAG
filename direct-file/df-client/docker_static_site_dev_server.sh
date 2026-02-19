#!/usr/bin/env bash

set -e

# Run development server in a container

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR"

docker buildx build \
  --pull \
  --file Dockerfile-static-site-development \
  --tag df-screener:dev \
  .

docker run \
  --rm \
  --volume ./packages:/build/packages \
  --volume ./tsconfig.base.json:/build/tsconfig.base.json \
  --volume ./eslint-plugin-df-rules/tsconfig.json:/build/eslint-plugin-df-rules/tsconfig.json \
  --volume ./eslint-plugin-df-rules/src:/build/eslint-plugin-df-rules/src \
  --volume ./js-factgraph-scala/src:/build/js-factgraph-scala/src \
  --volume ./df-static-site/.env.production:/build/df-static-site/.env.production \
  --volume ./df-static-site/tsconfig.json:/build/df-static-site/tsconfig.json \
  --volume ./df-static-site/vite.config.ts:/build/df-static-site/vite.config.ts \
  --volume ./df-static-site/index.html:/build/df-static-site/index.html \
  --volume ./df-static-site/public:/build/df-static-site/public \
  --volume ./df-static-site/src:/build/df-static-site/src \
  --publish "${DF_LISTEN_ADDRESS:-127.0.0.1}:${DF_SCREENER_PORT:-3500}:3500" \
    --name df-screener-development-server \
    --interactive \
    --tty \
    "$@" \
    df-screener:dev
