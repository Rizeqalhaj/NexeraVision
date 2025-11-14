#!/bin/bash
# NexaraVision Manual Setup for Vast.ai Web Terminal
# Copy and paste this entire script into the Vast.ai web terminal

echo "========================================"
echo "NexaraVision Manual Setup - Starting..."
echo "========================================"

# Create workspace
cd / && mkdir -p /workspace && cd /workspace

# Update system
apt-get update -qq && apt-get upgrade -y -qq

# Install essentials
apt-get install -y python3-pip git wget curl unzip ffmpeg aria2 tree

# Install Python packages
pip3 install --quiet tensorflow==2.13.0 opencv-python numpy pandas scikit-learn matplotlib kaggle tqdm

# Create directory structure
mkdir -p datasets/{tier1,tier2,tier3,processed} models logs checkpoints

# Configure Kaggle
mkdir -p ~/.kaggle
echo '{"username":"issadalu","key":"5aabafacbfdefea1bf4f2171d98cc52b"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Test setup
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import tensorflow as tf; print(f'GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"

echo ""
echo "✅ Setup complete! Now download the dataset script..."
echo ""
echo "Run this command to create the download script:"
cat << 'SCRIPT_END'

cat > /workspace/download_datasets.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path

# Dataset configuration
DATASETS = [
    ('vulamnguyen/rwf2000', 'tier1/RWF2000', 'RWF-2000'),
    ('odins0n/ucf-crime-dataset', 'tier1/UCF_Crime', 'UCF-Crime'),
    ('toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd', 'tier1/SCVD', 'SmartCity-CCTV'),
    ('mohamedmustafa/real-life-violence-situations-dataset', 'tier1/RealLifeViolence', 'Real-Life Violence'),
]

for kaggle_id, output_path, name in DATASETS:
    print(f"\n{'='*80}")
    print(f"Downloading: {name}")
    print(f"{'='*80}")

    out_dir = Path(f"/workspace/datasets/{output_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ['kaggle', 'datasets', 'download', '-d', kaggle_id, '-p', str(out_dir), '--unzip']

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {name} downloaded successfully!")
    except:
        print(f"❌ {name} failed to download")

print("\n" + "="*80)
print("Download Complete!")
print("="*80)
EOF

chmod +x /workspace/download_datasets.py

SCRIPT_END

echo ""
echo "After creating the script, run:"
echo "  python3 /workspace/download_datasets.py"
echo ""
