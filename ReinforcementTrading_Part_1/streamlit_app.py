from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Streamlit executes imported module top-level code, so this file remains the
# simple entrypoint users already know: `python -m streamlit run streamlit_app.py`.
from trading_bot.ui import app  # noqa: F401,E402
