#!/usr/bin/env python3
"""Inspect model architecture"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import h5py
import json

model_path = '/workspace/violence_detection_mvp/models/best_model.h5'

print("=" * 80)
print(f"📋 INSPECTING MODEL: {model_path}")
print("=" * 80 + "\n")

with h5py.File(model_path, 'r') as f:
    # Check top-level structure
    print("📁 Top-level keys:")
    for key in f.keys():
        print(f"   - {key}")

    # Get model config
    if 'model_config' in f.attrs:
        config_str = f.attrs['model_config']
        # Handle both string and bytes
        if isinstance(config_str, bytes):
            config_str = config_str.decode('utf-8')
        config = json.loads(config_str)
        print(f"\n📐 Model Class: {config.get('class_name', 'Unknown')}")
        print(f"📐 Backend: {config.get('backend', 'Unknown')}")

        # Get layers
        if 'config' in config:
            if 'layers' in config['config']:
                layers = config['config']['layers']
                print(f"\n🔧 Layers ({len(layers)} total):")
                for i, layer in enumerate(layers):
                    layer_class = layer.get('class_name', 'Unknown')
                    layer_name = layer.get('config', {}).get('name', 'unnamed')
                    print(f"   {i+1}. {layer_class:<20} (name: {layer_name})")

    # Check saved weights structure
    if 'model_weights' in f:
        print(f"\n💾 Saved Weight Groups:")
        weight_groups = list(f['model_weights'].keys())
        print(f"   Total groups: {len(weight_groups)}")
        for group in weight_groups[:10]:  # Show first 10
            print(f"   - {group}")
        if len(weight_groups) > 10:
            print(f"   ... and {len(weight_groups) - 10} more")

print("\n" + "=" * 80)
