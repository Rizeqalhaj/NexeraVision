#!/usr/bin/env python3
"""
Quick script to upload and run optimization on Vast.ai instance
Run this in your Jupyter terminal after uploading the files
"""

import os
import sys

print("=" * 80)
print("NexaraVision Optimization Runner")
print("=" * 80)

# Check if we're on the Vast.ai instance
if not os.path.exists('/workspace'):
    print("\n❌ ERROR: Not on Vast.ai instance!")
    print("   This script should run in your Jupyter terminal")
    print("   Current directory:", os.getcwd())
    sys.exit(1)

print("\n✅ Running on Vast.ai instance")
print(f"   Current directory: {os.getcwd()}")

# Check if scripts exist
required_files = [
    '/workspace/extract_frames_parallel.py',
    '/workspace/train_model_optimized.py'
]

missing_files = []
for f in required_files:
    if not os.path.exists(f):
        missing_files.append(f)

if missing_files:
    print("\n⚠️  Missing files:")
    for f in missing_files:
        print(f"   - {f}")
    print("\n📋 Please upload these files first:")
    print("   In Jupyter: Upload → /workspace/")
    sys.exit(1)

print("\n✅ All required files found!")

# Check if extraction already done
if os.path.exists('/workspace/processed/frames/extraction_metadata.json'):
    print("\n⚠️  Frames already extracted!")
    print("   Skipping extraction, proceeding to training...")
    import subprocess
    subprocess.run(['python3', '/workspace/train_model_optimized.py'])
else:
    print("\n🚀 Starting parallel frame extraction...")
    print("   This will use all 44 CPU cores!")
    print("   Expected time: 1-2 hours")
    print("\n" + "=" * 80)

    import subprocess

    # Run extraction
    result = subprocess.run(['python3', '/workspace/extract_frames_parallel.py'])

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ Extraction complete!")
        print("=" * 80)

        # Ask user if they want to start training immediately
        print("\n🚀 Ready to start optimized training!")
        print("   Run: python3 /workspace/train_model_optimized.py")
    else:
        print("\n❌ Extraction failed!")
        sys.exit(1)
