#!/usr/bin/env bash
#
# NeuroPredict — one-command launcher (macOS / Linux)
#
# Run it with:   ./run.sh
# (first time only:  chmod +x run.sh)
#
# This sets up everything by itself and then opens the website:
#   1. creates a private Python environment (.venv)
#   2. installs the libraries the app needs
#   3. trains the small demo model the first time (a few minutes, CPU only)
#   4. starts the website at http://localhost:8000 and opens your browser
#
# It works completely offline once the libraries are installed — no account,
# no token, and no Devin required.

set -euo pipefail

# Always run from the folder this script lives in.
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
URL="http://localhost:${PORT}"

# 1. Find a Python interpreter.
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "ERROR: Python 3 is not installed. Install it from https://www.python.org/downloads/ and run this again."
  exit 1
fi

# 2. Create the private environment the first time.
if [ ! -d ".venv" ]; then
  echo "==> Creating Python environment (.venv) ..."
  "$PY" -m venv .venv
fi

# Use the environment's python/pip directly (no need to 'activate').
VENV_PY=".venv/bin/python"

echo "==> Installing libraries (first run can take a few minutes) ..."
"$VENV_PY" -m pip install --quiet --upgrade pip
"$VENV_PY" -m pip install --quiet -r requirements.txt

# 3. Train the demo model the first time (only if it's missing).
if [ ! -f "models/wmd_multimodal.pt" ]; then
  echo "==> Training the demo model (first run only, a few minutes) ..."
  "$VENV_PY" scripts/train_demo.py
else
  echo "==> Demo model already present, skipping training."
fi

# 4. Start the website and open the browser.
echo ""
echo "==> Starting NeuroPredict at ${URL}"
echo "    (leave this window open while you use the site; press Ctrl+C to stop)"
echo ""

# Try to open the browser automatically, a moment after the server starts.
( sleep 3
  if command -v open >/dev/null 2>&1; then open "$URL"          # macOS
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" # Linux
  fi ) >/dev/null 2>&1 &

exec "$VENV_PY" -m uvicorn webapp.main:app --host 0.0.0.0 --port "$PORT"
