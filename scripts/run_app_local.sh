#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python 3.11-3.13 is required. Set PYTHON to its executable." >&2; exit 1
fi
if ! "$python_bin" -c 'import sys; raise SystemExit(not ((3,11) <= sys.version_info[:2] < (3,14)))'; then
  echo "Unsupported Python version; install Python 3.11, 3.12, or 3.13." >&2; exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ then rerun." >&2; exit 1
fi
echo "Syncing the locked local application environment (no destructive global changes)…"
uv sync --frozen --extra app --extra semantic --extra pdf
echo "Opening the private local application at http://127.0.0.1:8050"
exec uv run chatgpt-archive app --host 127.0.0.1 --port 8050 --output-directory archive-output
