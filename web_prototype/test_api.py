#!/usr/bin/env python3
"""
Nexara Vision Enterprise API Testing Script
Tests all endpoints and multi-model functionality
"""

import requests
import json
import sys
from pathlib import Path
import time

# Configuration
API_BASE_URL = "http://31.57.166.18:8005"
TEST_VIDEO_PATH = None  # Will search for test video

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")

def test_health():
    """Test health endpoint"""
    print_header("Testing Health Endpoint")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_success("Health check passed")
            print(json.dumps(data, indent=2))
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False

def test_models():
    """Test models endpoint"""
    print_header("Testing Models Endpoint")
    try:
        response = requests.get(f"{API_BASE_URL}/api/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_success("Models endpoint working")
            print(json.dumps(data, indent=2))

            # Check available models
            total = data.get('total_available', 0)
            if total == 5:
                print_success(f"All 5 models available")
            elif total > 0:
                print_info(f"{total} models available (expected 5)")
            else:
                print_error("No models available!")
                return False

            return True
        else:
            print_error(f"Models check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Models check error: {e}")
        return False

def test_info():
    """Test info endpoint"""
    print_header("Testing Info Endpoint")
    try:
        response = requests.get(f"{API_BASE_URL}/api/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_success("Info endpoint working")
            print(json.dumps(data, indent=2))
            return True
        else:
            print_error(f"Info check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Info check error: {e}")
        return False

def test_stats():
    """Test stats endpoint"""
    print_header("Testing Stats Endpoint")
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_success("Stats endpoint working")
            print(json.dumps(data, indent=2))
            return True
        else:
            print_error(f"Stats check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Stats check error: {e}")
        return False

def find_test_video():
    """Find a test video file"""
    search_paths = [
        "/home/admin/Desktop/NexaraVision/violence_detection_mvp/data",
        "/home/admin/Desktop/NexaraVision/datasets",
        "/home/admin/Desktop/NexaraVision",
        "/tmp"
    ]

    for search_path in search_paths:
        path = Path(search_path)
        if path.exists():
            for ext in ['.mp4', '.avi', '.mov']:
                videos = list(path.rglob(f'*{ext}'))
                if videos:
                    # Filter out very large files
                    for video in videos:
                        if video.stat().st_size < 50 * 1024 * 1024:  # < 50MB
                            return video
    return None

def test_upload(model_name=None):
    """Test video upload endpoint"""
    if model_name:
        print_header(f"Testing Upload Endpoint (Model: {model_name})")
    else:
        print_header("Testing Upload Endpoint (Default Model)")

    # Find test video
    video_path = find_test_video()
    if video_path is None:
        print_info("No test video found - skipping upload test")
        return True

    print_info(f"Using test video: {video_path}")

    try:
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            data = {}
            if model_name:
                data['model_name'] = model_name

            print_info("Uploading video...")
            start_time = time.time()

            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=files,
                data=data,
                timeout=120
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                print_success(f"Upload successful (took {elapsed:.2f}s)")
                print(json.dumps(result, indent=2))

                # Validate response structure
                required_fields = [
                    'violence_probability',
                    'non_violence_probability',
                    'classification',
                    'processing_time',
                    'model_used',
                    'frames_analyzed',
                    'confidence',
                    'timestamp'
                ]

                missing = [f for f in required_fields if f not in result]
                if missing:
                    print_error(f"Missing fields in response: {missing}")
                    return False

                # Check if model matches request
                if model_name and result.get('model_used') != model_name:
                    print_error(f"Model mismatch: requested {model_name}, got {result.get('model_used')}")
                    return False

                print_success(f"Classification: {result['classification']}")
                print_success(f"Model used: {result['model_used']}")
                print_success(f"Processing time: {result['processing_time']}s")
                print_success(f"Confidence: {result['confidence']:.2f}%")

                return True
            else:
                print_error(f"Upload failed: {response.status_code}")
                print(response.text)
                return False
    except Exception as e:
        print_error(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_models():
    """Test all available models"""
    print_header("Testing All Models")

    models = [
        'best_model',
        'ultimate_best_model',
        'ensemble_m1_best',
        'ensemble_m2_best',
        'ensemble_m3_best'
    ]

    results = {}
    for model in models:
        success = test_upload(model)
        results[model] = success
        time.sleep(2)  # Brief pause between tests

    print_header("Multi-Model Test Results")
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {model}")

    return all(results.values())

def main():
    """Run all tests"""
    print_header("Nexara Vision Enterprise API Test Suite")
    print(f"API Base URL: {API_BASE_URL}")
    print("")

    results = {}

    # Basic endpoint tests
    results['health'] = test_health()
    time.sleep(1)

    results['models'] = test_models()
    time.sleep(1)

    results['info'] = test_info()
    time.sleep(1)

    results['stats'] = test_stats()
    time.sleep(1)

    # Upload test with default model
    results['upload_default'] = test_upload()
    time.sleep(2)

    # Multi-model test
    if find_test_video():
        results['all_models'] = test_all_models()
    else:
        print_info("Skipping multi-model test - no test video available")
        results['all_models'] = True

    # Summary
    print_header("Test Summary")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")

    print("")
    print(f"Total: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print_success("All tests passed! 🎉")
        return 0
    else:
        print_error(f"{failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
