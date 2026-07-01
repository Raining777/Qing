#!/bin/bash
echo "============================================"
echo "  清 (Qing) — AI Study Assistant"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python not found. Please install Python 3.10+"
    exit 1
fi

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "[1/3] First time setup — creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "[2/3] Installing dependencies..."
pip install -r requirements.txt -q

# Check .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[INFO] Created .env from .env.example"
fi

# Suppress harmless TF warnings
export TF_ENABLE_ONEDNN_OPTS=0

# Start
echo "[3/3] Starting 清 (Qing)..."
echo ""
echo "   Opening http://localhost:7860"
echo "   Press Ctrl+C to stop"
echo ""

# Try to open browser
if command -v open &>/dev/null; then
    open http://localhost:7860
elif command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:7860
fi

python -m app.main
