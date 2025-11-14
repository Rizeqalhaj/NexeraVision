#!/usr/bin/env python3
"""
Advanced checkpoint inspector for Keras 3.x format
"""

import h5py
import json
from pathlib import Path

checkpoint_path = 'checkpoints/ultimate_best_model.h5'

if not Path(checkpoint_path).exists():
    print(f"Checkpoint not found: {checkpoint_path}")
    exit(1)

print("="*80)
print(f"INSPECTING: {checkpoint_path}")
print("="*80)

with h5py.File(checkpoint_path, 'r') as f:
    print("\n📦 Top-level structure:")

    def print_structure(name, obj, indent=0):
        """Recursively print HDF5 structure"""
        prefix = "  " * indent
        if isinstance(obj, h5py.Group):
            print(f"{prefix}📁 {name}/ ({len(obj)} items)")
            if len(obj) <= 20:  # Only expand small groups
                for key in list(obj.keys())[:10]:
                    print_structure(key, obj[key], indent + 1)
                if len(obj) > 10:
                    print(f"{prefix}  ... and {len(obj) - 10} more")
        elif isinstance(obj, h5py.Dataset):
            print(f"{prefix}📄 {name}: {obj.shape} {obj.dtype}")

    for key in f.keys():
        print_structure(key, f[key], 0)

    # Check for model weights structure
    print("\n" + "="*80)
    print("ANALYZING MODEL WEIGHTS")
    print("="*80)

    if 'model_weights' in f:
        mw = f['model_weights']

        # Keras 3.x stores weights differently
        print(f"\nModel weights structure: {type(mw)}")
        print(f"Keys in model_weights: {list(mw.keys())}")

        # Try to find layer information
        all_weights = []

        def collect_weights(name, obj):
            if isinstance(obj, h5py.Dataset):
                all_weights.append((name, obj.shape, obj.dtype))

        mw.visititems(collect_weights)

        print(f"\n📊 Found {len(all_weights)} weight tensors:")
        for name, shape, dtype in all_weights[:30]:  # Show first 30
            print(f"  {name:60s} {str(shape):20s} {dtype}")

        if len(all_weights) > 30:
            print(f"  ... and {len(all_weights) - 30} more")

        # Try to infer layer structure
        print("\n" + "="*80)
        print("LAYER STRUCTURE INFERENCE")
        print("="*80)

        layer_groups = {}
        for name, shape, dtype in all_weights:
            parts = name.split('/')
            if len(parts) > 0:
                layer_name = parts[0]
                if layer_name not in layer_groups:
                    layer_groups[layer_name] = []
                layer_groups[layer_name].append((name, shape))

        print(f"\nFound {len(layer_groups)} layer groups:")
        for i, (layer_name, weights) in enumerate(sorted(layer_groups.items()), 1):
            print(f"\n{i:2d}. {layer_name}")
            for weight_name, shape in weights[:5]:
                short_name = weight_name.replace(f"{layer_name}/", "  ")
                print(f"    {short_name:50s} {shape}")
            if len(weights) > 5:
                print(f"    ... and {len(weights) - 5} more")

print("\n" + "="*80)
print("TRY LOADING WITH KERAS")
print("="*80)

print("\nAttempting to load model with TensorFlow/Keras...")

import tensorflow as tf

try:
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    print(f"\n✅ SUCCESS! Model loaded")
    print(f"   Name: {model.name}")
    print(f"   Parameters: {model.count_params():,}")
    print(f"\n📋 Model Summary:")
    model.summary()

    print("\n✅ Model can be loaded! Use boost_universal.py")

except Exception as e:
    print(f"\n❌ Failed to load: {e}")
    print("\nTrying with safe_mode=False...")

    try:
        model = tf.keras.models.load_model(checkpoint_path, compile=False, safe_mode=False)
        print(f"\n✅ SUCCESS with safe_mode=False!")
        print(f"   Name: {model.name}")
        print(f"   Parameters: {model.count_params():,}")
        model.summary()

        print("\n✅ Use boost_universal.py with safe_mode=False")

    except Exception as e2:
        print(f"\n❌ Still failed: {e2}")

print("\n" + "="*80)
