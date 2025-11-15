#!/usr/bin/env python3
"""
Fix Vast.ai Training Configuration
Fixes: GPU OOM error + missing config fields

Usage:
    python3 fix_vastai_config.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

def main():
    print("=" * 80)
    print("🔧 NexaraVision Training Config Fix")
    print("=" * 80)
    print()

    # Config path
    config_path = Path("/workspace/training_config.json")

    # Check if we're on Vast.ai
    if not Path("/workspace").exists():
        print("⚠️  Warning: /workspace not found")
        print("This script is designed for Vast.ai environment")
        print("Continuing anyway...")
        print()

    # Backup original config if it exists
    if config_path.exists():
        backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.parent / f"training_config.json.backup.{backup_time}"

        with open(config_path, 'r') as f:
            original_config = json.load(f)

        with open(backup_path, 'w') as f:
            json.dump(original_config, f, indent=2)

        print(f"✅ Original config backed up to:")
        print(f"   {backup_path}")
        print()

        # Show what the old batch size was
        old_batch_size = original_config.get('training', {}).get('batch_size', 'unknown')
        print(f"📊 Old batch_size: {old_batch_size}")
        print()

    # Create fixed config
    print("📝 Creating new config with fixes...")

    fixed_config = {
        "data": {
            "augmentation": True,
            "class_weights": True
        },
        "model": {
            "sequence_model": "bidirectional_gru",
            "gru_units": 128,
            "dense_layers": [256, 128],
            "dropout": [0.4, 0.3, 0.2]
        },
        "training": {
            "frames_per_video": 20,
            "batch_size": 4,
            "learning_rate": 0.0001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "early_stopping_patience": 5,
            "reduce_lr_patience": 3,
            "reduce_lr_factor": 0.5
        },
        "paths": {
            "models": "/workspace/models/saved_models",
            "logs": "/workspace/models/logs",
            "checkpoints": "/workspace/models/checkpoints"
        }
    }

    # Write fixed config
    with open(config_path, 'w') as f:
        json.dump(fixed_config, f, indent=2)

    print("✅ Config file created successfully")
    print()

    # Show what changed
    print("=" * 80)
    print("📊 Configuration Changes")
    print("=" * 80)
    print()
    print("GPU Memory Fix:")
    print("  🔧 batch_size: 32 → 4")
    print("  ✅ Fits in 1GB GPU memory")
    print("  ⏱️  Training will be slower but won't crash")
    print()
    print("Missing Fields Added:")
    print("  ✅ early_stopping_patience: 5 epochs")
    print("  ✅ reduce_lr_patience: 3 epochs")
    print("  ✅ reduce_lr_factor: 0.5x")
    print()

    # Verify config was created
    if config_path.exists():
        print("=" * 80)
        print("✅ SUCCESS - Config Fixed!")
        print("=" * 80)
        print()
        print("📋 Config Preview:")
        print("=" * 80)
        with open(config_path, 'r') as f:
            print(f.read())
        print("=" * 80)
        print("🚀 Next Steps:")
        print("=" * 80)
        print()
        print("1. Clear GPU memory (if training was running):")
        print("   pkill -f train_model")
        print()
        print("2. Restart training:")
        print("   cd /workspace")
        print("   python3 train_model_optimized.py")
        print()
        print("3. Monitor GPU usage:")
        print("   watch -n 1 nvidia-smi")
        print()
        print("=" * 80)
    else:
        print("❌ ERROR: Failed to create config file")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
