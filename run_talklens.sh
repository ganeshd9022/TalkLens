#!/bin/zsh
# TalkLens Run Script
# ───────────────────
# This script ensures the correct Python 3.9 venv is used and resolves 
# library conflicts on macOS.

# 1. Resolve library duplicate warnings on Mac
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONIOENCODING=utf-8

# 2. Get absolute path to the venv
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PYTHON"
    echo "Please ensure you are in the project root."
    exit 1
fi

# 3. Run Streamlit using the venv's streamlit module
echo "🚀 Starting TalkLens (Python 3.9)..."
"$VENV_PYTHON" -m streamlit run "$SCRIPT_DIR/talklens/app.py"
