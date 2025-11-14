#!/usr/bin/env python3
"""
Inspect checkpoint structure to understand the exact architecture
"""

import h5py
import numpy as np
from pathlib import Path

checkpoint_path = 'checkpoints/ultimate_best_model.h5'

if not Path(checkpoint_path).exists():
    print(f"Checkpoint not found: {checkpoint_path}")
    exit(1)

print("="*80)
print(f"INSPECTING: {checkpoint_path}")
print("="*80)

with h5py.File(checkpoint_path, 'r') as f:
    print("\n📦 Top-level keys:")
    for key in f.keys():
        print(f"  - {key}")

    if 'model_weights' in f:
        print("\n🏗️  Model Structure:")
        model_weights = f['model_weights']

        layer_names = []

        def visit_layers(name, obj):
            if isinstance(obj, h5py.Group):
                # Check if this is a layer group
                if 'kernel:0' in obj or 'bias:0' in obj or 'gamma:0' in obj or 'beta:0' in obj:
                    layer_names.append(name)

        model_weights.visititems(visit_layers)

        print(f"\n📊 Found {len(layer_names)} weight groups:")
        for i, layer_name in enumerate(layer_names, 1):
            print(f"  {i:2d}. {layer_name}")

            # Get layer details
            layer_group = model_weights[layer_name]
            weights_list = []
            for weight_name in layer_group.keys():
                weight_shape = layer_group[weight_name].shape
                weights_list.append(f"{weight_name}: {weight_shape}")

            for w in weights_list[:3]:  # Show first 3 weights
                print(f"      {w}")

    # Try to infer architecture
    print("\n🔍 Architecture Analysis:")

    if 'model_weights' in f:
        mw = f['model_weights']

        # Count different layer types
        lstm_count = sum(1 for name in layer_names if 'lstm' in name.lower() or 'bilstm' in name.lower())
        dense_count = sum(1 for name in layer_names if 'dense' in name.lower())
        bn_count = sum(1 for name in layer_names if 'batch_normalization' in name.lower() or 'bn' in name.lower())
        dropout_count = sum(1 for name in layer_names if 'dropout' in name.lower())

        print(f"  LSTM layers: {lstm_count}")
        print(f"  Dense layers: {dense_count}")
        print(f"  BatchNorm layers: {bn_count}")
        print(f"  Dropout layers: {dropout_count}")

        # Check for bidirectional
        has_bidirectional = any('bidirectional' in name.lower() or 'forward' in name.lower() or 'backward' in name.lower()
                               for name in layer_names)
        print(f"  Bidirectional: {has_bidirectional}")

    # Check if this is Keras 3 format
    if 'config' in f:
        print("\n📋 Model Config:")
        config = f['config']
        if isinstance(config, h5py.Dataset):
            import json
            config_str = config[()].decode('utf-8')
            config_dict = json.loads(config_str)

            if 'config' in config_dict and 'layers' in config_dict['config']:
                layers = config_dict['config']['layers']
                print(f"\n  Total layers in config: {len(layers)}")
                print("\n  Layer types:")
                for i, layer in enumerate(layers, 1):
                    layer_class = layer.get('class_name', 'Unknown')
                    layer_name = layer.get('config', {}).get('name', layer.get('name', 'unnamed'))
                    print(f"    {i:2d}. {layer_class:20s} ({layer_name})")

print("\n" + "="*80)
