#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating .venv because it does not exist yet..."
  python3 -m venv ".venv"
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

if ! python - <<'PY' >/dev/null 2>&1
import streamlit
import stable_baselines3
import pandas
PY
then
  echo "Installing dependencies from requirements.txt..."
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m streamlit run streamlit_app.py
