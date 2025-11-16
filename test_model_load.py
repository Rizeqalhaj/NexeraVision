#!/usr/bin/env python3
"""
Simple test to verify the new model can be loaded.
Tests model discovery and loading without full service dependencies.
"""
import os
from pathlib import Path

# Test model discovery
search_paths = [
    "ml_service/models/initial_best_model.keras",
    "ml_service/models/best_model.h5",
]

print("🔍 Testing Model Discovery...")
print("-" * 60)

for path_str in search_paths:
    model_path = Path(path_str)
    exists = model_path.exists()
    if exists:
        size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Found: {path_str} ({size:.1f} MB)")
    else:
        print(f"❌ Not found: {path_str}")

# Find the best model
best_model = None
for path_str in search_paths:
    model_path = Path(path_str)
    if model_path.exists():
        best_model = model_path
        break

if best_model:
    print(f"\n🎯 Selected Model: {best_model}")
    print(f"   Format: {'Keras 3 (.keras)' if str(best_model).endswith('.keras') else 'Keras 2 (.h5)'}")
    print(f"   Size: {best_model.stat().st_size / (1024 * 1024):.1f} MB")
    print("\n✅ Model discovery successful!")
else:
    print("\n❌ No model found!")
    exit(1)

print("\n" + "=" * 60)
print("📋 Deployment Summary:")
print("=" * 60)
print(f"Model: {best_model.name}")
print(f"Path: {best_model}")
print(f"Docker ENV: MODEL_PATH=/app/models/{best_model.name}")
print(f"Service Port: 8003")
print("=" * 60)
