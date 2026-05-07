#!/usr/bin/env bash
# Run both test suites: Python unittest + Node/jsdom DOM tests.
# Exits non-zero on any failure (CI-friendly).

set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "Could not find $PY — run 'python3 -m venv .venv && pip install -r requirements.txt' first." >&2
  exit 1
fi

echo "== Python tests =="
"$PY" -m unittest discover tests/ -v

if [ -d node_modules/jsdom ]; then
  echo ""
  echo "== Frontend DOM tests =="
  node tests/test_frontend_render.js
else
  echo ""
  echo "== Frontend DOM tests skipped =="
  echo "(run 'npm install --no-save jsdom' to enable)"
fi
