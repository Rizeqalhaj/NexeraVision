#!/usr/bin/env python3
"""Quick GPU check"""

import tensorflow as tf

print("=" * 80)
print("GPU DIAGNOSTICS")
print("=" * 80)

# TensorFlow version
print(f"\nTensorFlow Version: {tf.__version__}")

# CUDA availability
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# GPU devices
gpus = tf.config.list_physical_devices('GPU')
print(f"\nPhysical GPUs: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu}")

# Test GPU computation
if len(gpus) > 0:
    print("\n✅ GPUs detected!")
    print("Testing GPU computation...")

    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print("✅ GPU computation works!")
        print(c.numpy())
else:
    print("\n❌ No GPUs detected!")
    print("\nTroubleshooting:")
    print("1. Run: nvidia-smi")
    print("2. Check CUDA: ls /usr/local/cuda*/lib64/libcudnn*")
    print("3. Install: pip install tensorflow[and-cuda]")

print("=" * 80)
