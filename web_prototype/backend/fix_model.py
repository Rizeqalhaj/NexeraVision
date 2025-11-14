#!/usr/bin/env python3
"""
Model fixer for old Keras 2.3 models
This script inspects and fixes H5 model configs to work with modern TensorFlow
"""
import h5py
import json
import sys
from pathlib import Path


def find_dtype_policy(obj, path=""):
    """Recursively find dtype_policy in config"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if 'dtype' in key.lower() or 'policy' in key.lower():
                print(f"Found at {path}.{key}: {value}")
            find_dtype_policy(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            find_dtype_policy(item, f"{path}[{i}]")


def fix_model_config(model_path):
    """Fix model config for old Keras models"""
    print(f"Inspecting: {model_path}")

    with h5py.File(model_path, 'r') as f:
        if 'model_config' not in f.attrs:
            print("No model_config found")
            return False

        config_str = f.attrs['model_config']
        if isinstance(config_str, bytes):
            config_str = config_str.decode('utf-8')

        config = json.loads(config_str)

        # Search for dtype_policy
        print("\nSearching for dtype_policy...")
        find_dtype_policy(config)

        # Show top-level structure
        print(f"\nTop-level keys: {config.keys()}")
        if 'config' in config:
            print(f"config keys: {config['config'].keys()}")
            if 'layers' in config['config']:
                print(f"Number of layers: {len(config['config']['layers'])}")
                # Show first layer config
                if config['config']['layers']:
                    first_layer = config['config']['layers'][0]
                    print(f"First layer: {first_layer.get('class_name')}")
                    print(f"First layer config keys: {first_layer.get('config', {}).keys()}")
                    print(f"First layer config: {json.dumps(first_layer.get('config', {}), indent=2)}")

        return True


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/app/models/best_model.h5"
    fix_model_config(model_path)
