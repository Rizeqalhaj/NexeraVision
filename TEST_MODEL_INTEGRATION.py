#!/usr/bin/env python3
"""
Quick test to verify the violence detection model loads correctly.
Run this before starting services to ensure the model is compatible.
"""

import sys
import os

# Add ml_service to path
sys.path.insert(0, '/home/admin/Desktop/NexaraVision/ml_service')

def test_model_loading():
    """Test that the model loads and can perform inference."""
    print("=" * 50)
    print("NexaraVision Model Integration Test")
    print("=" * 50)

    # Step 1: Check TensorFlow
    print("\n[1/5] Checking TensorFlow installation...")
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"  ✗ TensorFlow not installed: {e}")
        print("  Run: pip install tensorflow==2.15.0")
        return False

    # Step 2: Check model file exists
    print("\n[2/5] Checking model file...")
    model_path = "/home/admin/Desktop/NexaraVision/ml_service/models/initial_best_model.keras"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  ✓ Model found: {model_path}")
        print(f"  ✓ Model size: {size_mb:.1f} MB")
    else:
        print(f"  ✗ Model not found: {model_path}")
        return False

    # Step 3: Load custom AttentionLayer
    print("\n[3/5] Loading custom AttentionLayer...")
    try:
        from app.models.violence_detector import AttentionLayer
        print("  ✓ AttentionLayer loaded")
    except ImportError as e:
        print(f"  ✗ Failed to load AttentionLayer: {e}")
        return False

    # Step 4: Load model
    print("\n[4/5] Loading Keras model...")
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'AttentionLayer': AttentionLayer},
            compile=False
        )
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Input shape: {model.input_shape}")
        print(f"  ✓ Output shape: {model.output_shape}")
        print(f"  ✓ Total layers: {len(model.layers)}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Test inference with dummy data
    print("\n[5/5] Testing inference with dummy data...")
    try:
        import numpy as np

        # Create dummy input: (1, 20, 224, 224, 3) - batch of 20 frames
        dummy_input = np.zeros((1, 20, 224, 224, 3), dtype=np.float32)

        # Run inference
        prediction = model.predict(dummy_input, verbose=0)

        print(f"  ✓ Inference successful")
        print(f"  ✓ Output shape: {prediction.shape}")
        print(f"  ✓ Sample prediction: {prediction[0]}")

        # Interpret results
        if prediction.shape[-1] == 2:
            non_violence_prob = float(prediction[0][0])
            violence_prob = float(prediction[0][1])
            print(f"\n  Violence Probability: {violence_prob:.4f}")
            print(f"  Non-Violence Probability: {non_violence_prob:.4f}")
        else:
            print(f"  ⚠ Unexpected output shape: {prediction.shape}")

    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("✓ All tests passed! Model is ready for production.")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Run: ./START_NEXARA_SERVICES.sh")
    print("  2. Open: http://localhost:8001/live")
    print("  3. Test file upload or live camera detection")

    return True


if __name__ == "__main__":
    os.chdir('/home/admin/Desktop/NexaraVision/ml_service')
    success = test_model_loading()
    sys.exit(0 if success else 1)
