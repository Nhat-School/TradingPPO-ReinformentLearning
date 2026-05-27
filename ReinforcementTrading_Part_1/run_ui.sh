#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

pick_python() {
  for candidate in python3.12 python3.11 python3.10 /usr/bin/python3 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      "$candidate" - <<'PY' >/dev/null 2>&1 && { echo "$candidate"; return 0; }
import sys
raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)
PY
    fi
  done
  return 1
}

PYTHON_BIN="${PYTHON_BIN:-$(pick_python)}"
if [ -z "$PYTHON_BIN" ]; then
  echo "Could not find Python 3.9-3.12. Please install Python 3.11 or 3.12 for this ML project."
  exit 1
fi

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ] && [ -d "venv" ]; then
  VENV_DIR="venv"
fi

if [ -d "$VENV_DIR" ] && ! "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)
PY
then
  echo "Existing $VENV_DIR uses unsupported Python; recreating a clean .venv with $PYTHON_BIN..."
  rm -rf ".venv"
  VENV_DIR=".venv"
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating $VENV_DIR with $PYTHON_BIN..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! python - <<'PY' >/dev/null 2>&1
import streamlit
import stable_baselines3
import pandas
PY
then
  if [ "$VENV_DIR" = "venv" ]; then
    echo "Existing venv is missing or has broken ML dependencies; creating a clean .venv..."
    deactivate || true
    VENV_DIR=".venv"
    rm -rf "$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  fi
  echo "Installing dependencies from requirements.txt..."
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Avoid looking at a stale Streamlit process on port 8501 after code changes.
pkill -f "streamlit.*streamlit_app.py" >/dev/null 2>&1 || true

python -m streamlit run streamlit_app.py --server.headless true --server.address localhost --server.port 8501
