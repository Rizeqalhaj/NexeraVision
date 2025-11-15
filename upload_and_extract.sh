#!/bin/bash
# Upload optimization scripts and run parallel extraction

echo "=================================================="
echo "NexaraVision Optimization - 44 CPU Core Extraction"
echo "=================================================="

# Upload the scripts (you'll need to copy these to the instance)
echo ""
echo "📋 Step 1: Stop the current training (if still running)"
echo "   Press Ctrl+C in the training terminal"
echo ""

echo "📋 Step 2: Upload these files to /workspace/:"
echo "   - extract_frames_parallel.py"
echo "   - train_model_optimized.py"
echo ""

echo "📋 Step 3: Run this command in your Jupyter terminal:"
echo ""
echo "   cd /workspace && python3 extract_frames_parallel.py"
echo ""

echo "⏱️  Expected extraction time: 1-2 hours"
echo "💾 Expected output size: ~15-20 GB"
echo "📁 Output location: /workspace/processed/frames/*.npy"
echo ""

echo "📋 Step 4: After extraction completes, run:"
echo ""
echo "   python3 train_model_optimized.py"
echo ""

echo "⏱️  Expected training time: 6-8 hours"
echo "💰 Total cost: ~$9 (vs $27 without optimization)"
echo "⚡ Speedup: 5-10x faster!"
echo "=================================================="
