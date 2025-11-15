#!/usr/bin/env python3
"""
Test NexaraVision Training Pipeline
Validates all components before full training
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

print("=" * 80)
print("NexaraVision Pipeline Validation")
print("=" * 80)

try:
    print("\n1️⃣ Testing imports...")
    from data_preprocessing import VideoDataPreprocessor
    from model_architecture import ViolenceDetectionModel
    print("   ✅ All imports successful")

    # ===================================================================
    # TEST 1: Data Preprocessing
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Data Preprocessing")
    print("=" * 80)

    preprocessor = VideoDataPreprocessor()
    total_videos = preprocessor.scan_datasets()

    if total_videos == 0:
        print("❌ No videos found!")
        sys.exit(1)

    print(f"✅ Found {total_videos:,} videos")

    # Test frame extraction on first video
    print("\n2️⃣ Testing frame extraction...")
    test_video = preprocessor.video_paths[0]
    print(f"   Test video: {Path(test_video).name}")

    frames = preprocessor.extract_frames(test_video)
    print(f"   ✅ Extracted frames shape: {frames.shape}")
    print(f"   ✅ Value range: [{frames.min():.3f}, {frames.max():.3f}]")

    assert frames.shape == (20, 224, 224, 3), "Frame shape incorrect!"
    assert frames.min() >= 0 and frames.max() <= 1, "Frame values not normalized!"

    # Test data splits
    print("\n3️⃣ Testing data splits...")
    splits = preprocessor.create_splits()
    print("   ✅ Splits created successfully")

    # ===================================================================
    # TEST 2: Model Architecture
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Model Architecture")
    print("=" * 80)

    print("\n4️⃣ Building model...")
    model_builder = ViolenceDetectionModel()
    model = model_builder.build_model(trainable_backbone=False)
    print("   ✅ Model built successfully")

    print("\n5️⃣ Compiling model...")
    model_builder.compile_model()
    print("   ✅ Model compiled successfully")

    print("\n6️⃣ Counting parameters...")
    trainable, non_trainable, total = model_builder.count_parameters()
    print(f"   ✅ Total parameters: {total:,}")

    # ===================================================================
    # TEST 3: Forward Pass
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Forward Pass")
    print("=" * 80)

    print("\n7️⃣ Testing forward pass with real frames...")
    # Use extracted frames from test video
    batch_frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    print(f"   Input shape: {batch_frames.shape}")

    output = model(batch_frames, training=False)
    print(f"   ✅ Output shape: {output.shape}")
    print(f"   ✅ Predictions: {output.numpy()}")

    predicted_class = tf.argmax(output, axis=1).numpy()[0]
    confidence = output.numpy()[0][predicted_class]
    print(f"   ✅ Predicted class: {predicted_class} (confidence: {confidence:.2%})")

    # ===================================================================
    # TEST 4: Data Generator
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Data Generator")
    print("=" * 80)

    print("\n8️⃣ Testing data generator...")
    from train_model import VideoDataGenerator

    X_train, y_train, _ = splits['train']

    # Create generator with small batch
    train_gen = VideoDataGenerator(
        X_train[:4],  # Just 4 videos for testing
        y_train[:4],
        preprocessor,
        batch_size=2,
        shuffle=False
    )

    print(f"   Generator length: {len(train_gen)}")

    # Get one batch
    X_batch, y_batch = train_gen[0]
    print(f"   ✅ Batch shapes: X={X_batch.shape}, y={y_batch.shape}")

    assert X_batch.shape == (2, 20, 224, 224, 3), "Batch X shape incorrect!"
    assert y_batch.shape == (2, 2), "Batch y shape incorrect!"

    # ===================================================================
    # TEST 5: Training Step
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Single Training Step")
    print("=" * 80)

    print("\n9️⃣ Testing single training step...")

    # Train on one batch
    history = model.fit(
        X_batch, y_batch,
        epochs=1,
        verbose=1
    )

    print("   ✅ Training step successful")
    print(f"   Loss: {history.history['loss'][0]:.4f}")
    print(f"   Accuracy: {history.history['accuracy'][0]:.4f}")

    # ===================================================================
    # TEST 6: GPU Verification
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 6: GPU Verification")
    print("=" * 80)

    print("\n🔟 Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"   GPUs detected: {len(gpus)}")

    for i, gpu in enumerate(gpus):
        print(f"   ✅ GPU {i}: {gpu.name}")

    if len(gpus) == 0:
        print("   ⚠️  No GPUs detected - training will be SLOW on CPU")
    else:
        print(f"   ✅ {len(gpus)} GPU(s) ready for training")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print("\n✅ ALL TESTS PASSED!")
    print("\nVerified Components:")
    print("  ✅ Data preprocessing pipeline")
    print("  ✅ Model architecture (ResNet50V2 + Bi-LSTM)")
    print("  ✅ Model compilation")
    print("  ✅ Forward pass")
    print("  ✅ Data generator")
    print("  ✅ Training step")
    print(f"  ✅ GPU availability ({len(gpus)} GPUs)")

    print("\n" + "=" * 80)
    print("🎉 SYSTEM READY FOR TRAINING!")
    print("=" * 80)

    print("\nTo start training, run:")
    print("  python3 train_model.py")

    print("\nExpected training time:")
    print("  Initial phase (30 epochs): ~3-5 hours")
    print("  Fine-tuning (20 epochs): ~2-3 hours")
    print("  Total: ~5-8 hours")

    print("\nMonitor training:")
    print("  TensorBoard: tensorboard --logdir /workspace/logs/training")
    print("  CSV logs: /workspace/logs/training/*.csv")

except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("   Make sure all dependencies are installed:")
    print("   pip install tensorflow opencv-python numpy pandas scikit-learn")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
