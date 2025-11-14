#!/usr/bin/env python3
"""
Enable Mixed Precision Training (FP16) to reduce GPU memory usage
This MIGHT work on 1GB GPU but is not guaranteed
"""

import json
from pathlib import Path

def enable_mixed_precision():
    """Add mixed precision config to training setup"""

    config_path = Path("/workspace/training_config.json")

    # Read existing config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Add mixed precision settings
    config['training']['mixed_precision'] = True
    config['training']['batch_size'] = 4  # Keep at minimum

    # Write updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 80)
    print("✅ Mixed Precision Enabled")
    print("=" * 80)
    print("\nConfig Updated:")
    print(f"  - mixed_precision: True")
    print(f"  - batch_size: 4")
    print("\n⚠️  WARNING: This may still fail on 1GB GPU")
    print("   Mixed precision reduces memory by ~40% but ResNet50V2 needs 1.2GB+")
    print("\n🚀 Next: Update train_model_optimized.py to use mixed precision")
    print("=" * 80)

if __name__ == "__main__":
    enable_mixed_precision()
