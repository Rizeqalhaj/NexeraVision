#!/bin/bash
# Install dependencies for train_HYBRID_OPTIMAL_from_scratch.py on Vast.ai

echo "Installing dependencies for HYBRID OPTIMAL training..."

# Install OpenCV and other required packages
pip install --break-system-packages opencv-python-headless tqdm

echo ""
echo "✅ Installation complete!"
echo ""
echo "Installed packages:"
echo "  - opencv-python-headless (cv2)"
echo "  - tqdm (progress bars)"
echo ""
echo "Now you can run:"
echo "  python3 train_HYBRID_OPTIMAL_from_scratch.py"
