#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
EXPECTED_VENV="$PROJECT_ROOT/.venv_ai"
EXPECTED_PY="$EXPECTED_VENV/bin/python"

echo "Project: $PROJECT_ROOT"
echo "Expected venv: $EXPECTED_VENV"

if [[ "${VIRTUAL_ENV:-}" != "$EXPECTED_VENV" ]]; then
  echo
  echo "WRONG ENV DETECTED"
  echo "ENV CHECK: FAIL"
  echo "Active VIRTUAL_ENV is not .venv_ai"
  echo "Current: ${VIRTUAL_ENV:-<none>}"
  echo
  echo "Activate with:"
  echo "source \"$EXPECTED_VENV/bin/activate\""
  exit 1
fi

if [[ ! -x "$EXPECTED_PY" ]]; then
  echo
  echo "ENV CHECK: FAIL"
  echo "Missing expected python executable:"
  echo "$EXPECTED_PY"
  exit 1
fi

ACTIVE_PY="$(command -v python || true)"
ACTIVE_VER="$(python --version 2>&1 || true)"

echo
echo "ENV CHECK: OK"
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "python: $ACTIVE_PY"
echo "python --version: $ACTIVE_VER"

if [[ "$ACTIVE_PY" != "$EXPECTED_PY" ]]; then
  echo
  echo "WARNING: 'python' does not resolve to .venv_ai/bin/python"
  echo "Expected: $EXPECTED_PY"
  echo "Got:      $ACTIVE_PY"
  echo "If needed run: hash -r"
fi
