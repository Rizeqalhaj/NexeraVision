"""
Batch Testing Script for Grid Detection

Tests detector on multiple CCTV screenshots to validate
research-predicted 80-91% success rate
"""

import os
import cv2
import json
from pathlib import Path
from detector import GridDetector
from typing import List, Dict


def batch_test(
    image_dir: str,
    output_dir: str = "test_results",
    min_confidence: float = 0.7
) -> Dict:
    """
    Test grid detector on all images in directory

    Args:
        image_dir: Directory containing CCTV screenshots
        output_dir: Directory to save results and visualizations
        min_confidence: Minimum confidence threshold

    Returns:
        Summary statistics dictionary
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize detector
    detector = GridDetector(min_confidence=min_confidence)

    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Collect all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"No images found in {image_dir}")
        return {}

    print(f"Found {len(image_files)} images to test\n")
    print("=" * 80)

    # Test results
    results = []
    success_count = 0
    manual_required_count = 0
    failed_count = 0

    # Test each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Testing: {image_path.name}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  ❌ Error: Could not load image")
            failed_count += 1
            continue

        # Run detection
        result = detector.detect(image)

        # Visualize and save
        vis_path = os.path.join(output_dir, f"{image_path.stem}_detected.jpg")
        detector.visualize_detection(image, result, vis_path)

        # Collect statistics
        if result['success']:
            success_count += 1
            status = "✅ SUCCESS"
        elif result['confidence'] > 0.5:
            manual_required_count += 1
            status = "⚠️  MANUAL REQUIRED"
        else:
            failed_count += 1
            status = "❌ FAILED"

        print(f"  {status}")
        print(f"  Grid: {result['grid_layout'][0]}×{result['grid_layout'][1]}")
        print(f"  Regions: {len(result['regions'])}")
        print(f"  Confidence: {result['confidence']:.2%}")

        # Save result
        results.append({
            'filename': image_path.name,
            'success': result['success'],
            'grid_layout': result['grid_layout'],
            'regions_count': len(result['regions']),
            'confidence': result['confidence'],
            'requires_manual': result['requires_manual'],
            'visualization': vis_path
        })

    # Calculate statistics
    total = len(image_files)
    success_rate = (success_count / total * 100) if total > 0 else 0
    manual_rate = (manual_required_count / total * 100) if total > 0 else 0
    failure_rate = (failed_count / total * 100) if total > 0 else 0

    summary = {
        'total_images': total,
        'successful': success_count,
        'manual_required': manual_required_count,
        'failed': failed_count,
        'success_rate': success_rate,
        'manual_rate': manual_rate,
        'failure_rate': failure_rate,
        'results': results
    }

    # Print summary
    print("\n" + "=" * 80)
    print("BATCH TEST SUMMARY")
    print("=" * 80)
    print(f"Total Images: {total}")
    print(f"✅ Successful: {success_count} ({success_rate:.1f}%)")
    print(f"⚠️  Manual Required: {manual_required_count} ({manual_rate:.1f}%)")
    print(f"❌ Failed: {failed_count} ({failure_rate:.1f}%)")
    print("=" * 80)

    # Research validation
    print("\nResearch Validation:")
    print(f"Expected Success Rate: 80-91%")
    print(f"Actual Success Rate: {success_rate:.1f}%")

    if 80 <= success_rate <= 91:
        print("✅ WITHIN RESEARCH-PREDICTED RANGE")
    elif success_rate > 91:
        print("🎉 EXCEEDS RESEARCH PREDICTIONS")
    else:
        print("⚠️  BELOW RESEARCH PREDICTIONS - Consider parameter tuning")

    # Save results to JSON
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: {output_dir}/")

    return summary


def analyze_failures(results_file: str):
    """
    Analyze failure patterns from batch test results

    Args:
        results_file: Path to results.json file
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    print("\nFAILURE ANALYSIS")
    print("=" * 80)

    # Find low-confidence detections
    low_confidence = [r for r in data['results'] if r['confidence'] < 0.7]

    if not low_confidence:
        print("No failures to analyze!")
        return

    print(f"Found {len(low_confidence)} low-confidence detections:\n")

    for result in low_confidence:
        print(f"File: {result['filename']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Grid: {result['grid_layout']}")
        print(f"  Regions: {result['regions_count']}")
        print()

    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("1. Manually inspect visualizations for failed cases")
    print("2. Identify common failure patterns (overlays, contrast, layout)")
    print("3. Consider parameter tuning for specific CCTV brands")
    print("4. Use manual calibration fallback for challenging cases")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python batch_test.py <image_directory>")
        print("  python batch_test.py analyze <results.json>")
        sys.exit(1)

    if sys.argv[1] == "analyze":
        if len(sys.argv) < 3:
            print("Usage: python batch_test.py analyze <results.json>")
            sys.exit(1)
        analyze_failures(sys.argv[2])
    else:
        image_dir = sys.argv[1]
        if not os.path.isdir(image_dir):
            print(f"Error: {image_dir} is not a directory")
            sys.exit(1)

        batch_test(image_dir)
