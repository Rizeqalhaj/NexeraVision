#!/usr/bin/env python3
"""
NexaraVision ML Service Setup Verification

Checks all prerequisites and configuration before starting the service.
"""
import sys
from pathlib import Path
import subprocess


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def check_python_version():
    """Check Python version >= 3.10."""
    print_section("Python Version")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 10:
        print("✓ Python version OK")
        return True
    else:
        print(f"✗ Python 3.10+ required, found {version.major}.{version.minor}")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print_section("Python Dependencies")

    required_packages = [
        'fastapi',
        'uvicorn',
        'tensorflow',
        'cv2',
        'numpy',
        'pydantic'
    ]

    all_installed = True
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✓ opencv-python: {cv2.__version__}")
            elif package == 'tensorflow':
                import tensorflow as tf
                print(f"✓ tensorflow: {tf.__version__}")
            elif package == 'numpy':
                import numpy as np
                print(f"✓ numpy: {np.__version__}")
            elif package == 'fastapi':
                import fastapi
                print(f"✓ fastapi: {fastapi.__version__}")
            elif package == 'uvicorn':
                import uvicorn
                print(f"✓ uvicorn: {uvicorn.__version__}")
            elif package == 'pydantic':
                import pydantic
                print(f"✓ pydantic: {pydantic.__version__}")
        except ImportError:
            print(f"✗ {package} not installed")
            all_installed = False

    return all_installed


def check_model_file():
    """Check if model file exists."""
    print_section("Model File")

    model_paths = [
        Path("models/ultimate_best_model.h5"),
        Path("../downloaded_models/ultimate_best_model.h5"),
    ]

    for model_path in model_paths:
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✓ Model found: {model_path}")
            print(f"  Size: {size_mb:.1f} MB")
            return True

    print("✗ Model file not found")
    print("\nExpected locations:")
    for path in model_paths:
        print(f"  - {path.absolute()}")

    print("\nTo fix:")
    print("  mkdir -p models")
    print("  cp ../downloaded_models/ultimate_best_model.h5 models/")

    return False


def check_gpu():
    """Check GPU availability."""
    print_section("GPU Configuration")

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            print(f"✓ {len(gpus)} GPU(s) detected:")
            for gpu in gpus:
                print(f"  - {gpu.name}")

            # Test GPU
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                c = tf.matmul(a, b)
                print(f"✓ GPU test passed: {c.shape}")

            return True
        else:
            print("⚠ No GPU detected - will use CPU")
            print("  Service will work but inference will be slower")
            return True  # Not a failure

    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        return False


def check_nvidia_driver():
    """Check NVIDIA driver installation."""
    print_section("NVIDIA Driver")

    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            # Parse driver version
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version:' in line:
                    print(f"✓ {line.strip()}")
                    break
            return True
        else:
            print("✗ nvidia-smi failed")
            return False

    except FileNotFoundError:
        print("⚠ nvidia-smi not found - GPU may not be available")
        return True  # Not a critical failure
    except Exception as e:
        print(f"⚠ Could not check NVIDIA driver: {e}")
        return True  # Not a critical failure


def check_port():
    """Check if port 8000 is available."""
    print_section("Port Availability")

    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()

        if result != 0:
            print("✓ Port 8000 is available")
            return True
        else:
            print("⚠ Port 8000 is already in use")
            print("  Set PORT environment variable to use different port")
            print("  export PORT=8001")
            return True  # Not a critical failure

    except Exception as e:
        print(f"⚠ Could not check port: {e}")
        return True


def check_directory_structure():
    """Check project directory structure."""
    print_section("Directory Structure")

    required_dirs = [
        'app',
        'app/api',
        'app/core',
        'app/models',
        'app/utils',
        'tests',
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ missing")
            all_exist = False

    return all_exist


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print(" NexaraVision ML Service - Setup Verification")
    print("="*60)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Model File", check_model_file),
        ("NVIDIA Driver", check_nvidia_driver),
        ("GPU Configuration", check_gpu),
        ("Port Availability", check_port),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name} check failed: {e}")
            results[name] = False

    # Summary
    print_section("Verification Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ All checks passed! Ready to start service.")
        print("\nTo start the service:")
        print("  ./start_service.sh")
        print("  OR")
        print("  python -m uvicorn app.main:app --reload")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
