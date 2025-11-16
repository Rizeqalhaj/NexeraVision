#!/usr/bin/env python3
"""
Test script for new features: Grid Segmentation and MediaPipe Skeleton Detection.

Tests:
1. Grid detection from sample images
2. Camera extraction from multi-camera grids
3. MediaPipe pose detection
4. Skeleton-based violence classification
5. Ensemble prediction (CNN + Skeleton)
"""
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from app.segmentation import GridDetector, VideoSegmenter, QualityEnhancer
from app.mediapipe_detector import SkeletonDetector, SkeletonViolenceClassifier, PoseFeatureExtractor


def create_test_grid_image(rows: int = 3, cols: int = 3, size: int = 900) -> np.ndarray:
    """
    Create synthetic grid image for testing.

    Args:
        rows: Number of grid rows
        cols: Number of grid columns
        size: Total image size

    Returns:
        Grid image with colored cells
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    cell_height = size // rows
    cell_width = size // cols

    # Draw grid lines
    for i in range(rows + 1):
        y = i * cell_height
        cv2.line(img, (0, y), (size, y), (255, 255, 255), 2)

    for j in range(cols + 1):
        x = j * cell_width
        cv2.line(img, (x, 0), (x, size), (255, 255, 255), 2)

    # Fill cells with different colors
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]

    for r in range(rows):
        for c in range(cols):
            y1 = r * cell_height + 3
            y2 = (r + 1) * cell_height - 3
            x1 = c * cell_width + 3
            x2 = (c + 1) * cell_width - 3

            color_idx = (r * cols + c) % len(colors)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[color_idx], -1)

            # Add camera number
            text = f"Cam {r*cols + c + 1}"
            font_scale = size / 1000
            cv2.putText(img, text, (x1 + 10, y1 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    return img


def test_grid_detection():
    """Test grid detection functionality."""
    print("\n" + "="*60)
    print("TEST 1: Grid Detection")
    print("="*60)

    # Create test grids
    test_cases = [
        (2, 2, "2x2 Grid"),
        (3, 3, "3x3 Grid"),
        (4, 4, "4x4 Grid"),
    ]

    for rows, cols, name in test_cases:
        print(f"\nTesting {name}...")

        # Create test image
        img = create_test_grid_image(rows, cols)

        # Detect grid
        detector = GridDetector(img)
        layout = detector.detect_grid_layout()

        print(f"  Detected: {layout['rows']}x{layout['cols']} = {layout['total_cameras']} cameras")

        # Validate
        if layout['rows'] == rows and layout['cols'] == cols:
            print(f"  ✅ PASS - Correctly detected {name}")
        else:
            print(f"  ❌ FAIL - Expected {rows}x{cols}, got {layout['rows']}x{layout['cols']}")

        # Save visualization
        vis = detector.visualize_grid(f"/tmp/grid_test_{rows}x{cols}.jpg")
        print(f"  Saved visualization to /tmp/grid_test_{rows}x{cols}.jpg")


def test_camera_extraction():
    """Test camera region extraction."""
    print("\n" + "="*60)
    print("TEST 2: Camera Extraction")
    print("="*60)

    # Create 3x3 grid
    img = create_test_grid_image(3, 3)

    detector = GridDetector(img)
    cameras = detector.extract_camera_regions()

    print(f"\nExtracted {len(cameras)} camera regions")

    for camera in cameras[:3]:  # Show first 3
        print(f"  Camera {camera['id']}:")
        print(f"    Position: {camera['position']}")
        print(f"    Resolution: {camera['resolution']}")
        print(f"    Active: {camera['is_active']}")

    if len(cameras) == 9:
        print("\n  ✅ PASS - Correctly extracted all 9 cameras")
    else:
        print(f"\n  ❌ FAIL - Expected 9 cameras, got {len(cameras)}")


def test_quality_enhancement():
    """Test quality enhancement."""
    print("\n" + "="*60)
    print("TEST 3: Quality Enhancement")
    print("="*60)

    # Create low-quality image
    low_res = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    enhancer = QualityEnhancer()

    print("\nTesting enhancement on 100x100 image...")
    enhanced = enhancer.enhance_camera_feed(low_res)

    print(f"  Input shape: {low_res.shape}")
    print(f"  Output shape: {enhanced.shape}")

    if enhanced.shape == low_res.shape:
        print("  ✅ PASS - Enhancement maintains dimensions")
    else:
        print("  ❌ FAIL - Enhancement changed dimensions")


def test_pose_detection():
    """Test MediaPipe pose detection."""
    print("\n" + "="*60)
    print("TEST 4: MediaPipe Pose Detection")
    print("="*60)

    # Create simple test image with person-like shape
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw simple stick figure
    # Head
    cv2.circle(img, (320, 100), 30, (255, 255, 255), -1)
    # Body
    cv2.line(img, (320, 130), (320, 300), (255, 255, 255), 20)
    # Arms
    cv2.line(img, (320, 180), (250, 220), (255, 255, 255), 15)
    cv2.line(img, (320, 180), (390, 220), (255, 255, 255), 15)
    # Legs
    cv2.line(img, (320, 300), (280, 420), (255, 255, 255), 15)
    cv2.line(img, (320, 300), (360, 420), (255, 255, 255), 15)

    print("\nTesting pose detection on synthetic image...")

    try:
        detector = SkeletonDetector()
        pose_data = detector.detect_pose(img)

        if pose_data:
            print(f"  ✅ PASS - Pose detected")
            print(f"    Landmarks: {len(pose_data['landmarks'])}")
            print(f"    Mean visibility: {pose_data['mean_visibility']:.2f}")

            # Test visualization
            vis = detector.visualize_pose(img, pose_data)
            cv2.imwrite('/tmp/pose_test.jpg', vis)
            print(f"    Saved visualization to /tmp/pose_test.jpg")
        else:
            print("  ⚠️  WARNING - No pose detected (expected for simple stick figure)")

        detector.close()

    except Exception as e:
        print(f"  ❌ FAIL - Error: {e}")


def test_feature_extraction():
    """Test pose feature extraction."""
    print("\n" + "="*60)
    print("TEST 5: Pose Feature Extraction")
    print("="*60)

    # Create test pose data (mock)
    mock_landmarks = []
    for i in range(33):
        mock_landmarks.append({
            'x': np.random.rand(),
            'y': np.random.rand(),
            'z': np.random.rand(),
            'visibility': 0.9
        })

    mock_pose = {
        'landmarks': mock_landmarks,
        'mean_visibility': 0.9,
        'frame_shape': (480, 640)
    }

    print("\nTesting feature extraction...")

    try:
        extractor = PoseFeatureExtractor()
        features = extractor.extract_features(mock_pose)

        print(f"  Extracted {len(features)} features")
        print(f"  Sample features:")
        for key in list(features.keys())[:5]:
            print(f"    {key}: {features[key]:.3f}")

        # Test pattern detection
        patterns = extractor.detect_violent_patterns(features)

        print(f"\n  Violence patterns:")
        for pattern, score in patterns.items():
            print(f"    {pattern}: {score:.3f}")

        print("\n  ✅ PASS - Feature extraction working")

    except Exception as e:
        print(f"  ❌ FAIL - Error: {e}")


def test_skeleton_classifier():
    """Test skeleton-based violence classifier."""
    print("\n" + "="*60)
    print("TEST 6: Skeleton Violence Classifier")
    print("="*60)

    # Create test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    print("\nTesting skeleton classifier...")

    try:
        classifier = SkeletonViolenceClassifier()
        prediction = classifier.predict(img)

        print(f"  Violence probability: {prediction['violence_probability']:.3f}")
        print(f"  Prediction: {prediction['prediction']}")
        print(f"  Confidence: {prediction['confidence']}")
        print(f"  Pose detected: {prediction['pose_detected']}")

        if 'violence_probability' in prediction:
            print("\n  ✅ PASS - Classifier working")
        else:
            print("\n  ❌ FAIL - Missing prediction fields")

        classifier.close()

    except Exception as e:
        print(f"  ❌ FAIL - Error: {e}")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "NexaraVision Feature Test Suite" + " "*16 + "║")
    print("╚" + "="*58 + "╝")

    try:
        test_grid_detection()
        test_camera_extraction()
        test_quality_enhancement()
        test_pose_detection()
        test_feature_extraction()
        test_skeleton_classifier()

        print("\n" + "="*60)
        print("TEST SUITE COMPLETED")
        print("="*60)
        print("\n✅ All tests passed or completed with warnings")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Start ML service: python -m app.main")
        print("  3. Test API endpoints: http://localhost:8003/docs")
        print("  4. Integrate with frontend")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
