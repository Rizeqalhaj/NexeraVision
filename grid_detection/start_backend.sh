#!/bin/bash
# Start Grid Detection Backend

echo "Starting Grid Detection Backend on http://localhost:8004"
echo "=============================================="

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements (including FastAPI and Uvicorn)
echo "Installing requirements..."
pip install -q fastapi uvicorn opencv-python numpy

# Start FastAPI server on port 8004
echo "Starting FastAPI server..."
python3 -m uvicorn api_integration:app --host 0.0.0.0 --port 8004 --reload

echo "Backend started successfully on http://localhost:8004"
echo "API Docs: http://localhost:8004/docs"
