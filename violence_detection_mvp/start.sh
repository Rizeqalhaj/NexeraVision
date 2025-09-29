#!/bin/bash
# Violence Detection MVP - Quick Start Script

echo "🚀 Starting Violence Detection MVP..."
echo "=================================="

# Navigate to project directory
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
python -c "import cv2, tensorflow, numpy; print('✅ All dependencies available')" || {
    echo "❌ Dependencies missing! Installing..."
    pip install -r requirements.txt
}

echo ""
echo "✅ Setup complete! Available commands:"
echo ""
echo "  python run.py info           # Show system status"
echo "  python run.py --help         # Show all commands"
echo "  python test_model_architecture.py  # Test model (no data needed)"
echo "  python validate_implementation.py  # Validate system"
echo ""
echo "🎯 Quick test:"
echo "  python run.py info"
echo ""

# Keep shell open with venv activated
exec bash