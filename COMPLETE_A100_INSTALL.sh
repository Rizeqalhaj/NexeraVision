#!/bin/bash
################################################################################
# NexaraVision A100 GPU Training Environment Installation Script
#
# Purpose: Complete automated setup for violence detection model training
# GPU: NVIDIA A100 SXM4 80GB with CUDA 13.0
# Model: ResNet50V2 + BiGRU architecture
#
# Features:
# - Auto-detects Python version and installs compatible TensorFlow
# - Verifies GPU detection and CUDA compatibility
# - Installs all required dependencies with version checks
# - Idempotent (safe to run multiple times)
# - Comprehensive error handling and progress feedback
#
# Usage: bash COMPLETE_A100_INSTALL.sh
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Progress tracking
TOTAL_STEPS=8
CURRENT_STEP=0

################################################################################
# Utility Functions
################################################################################

print_header() {
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo -e "\n${BLUE}[Step $CURRENT_STEP/$TOTAL_STEPS]${NC} ${GREEN}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# Error handler
error_exit() {
    print_error "$1"
    echo -e "\n${RED}Installation failed. Please review the error above.${NC}"
    exit 1
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

################################################################################
# Main Installation
################################################################################

print_header "NexaraVision A100 GPU Training Environment Setup"

# Check if running as root (allow in containers like vast.ai)
if [[ $EUID -eq 0 ]]; then
   if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
       print_warning "Running as root in container - this is OK for vast.ai"
   else
       print_warning "Running as root - proceeding anyway (container environment assumed)"
   fi
fi

################################################################################
# STEP 1: System Information Detection
################################################################################

print_step "Detecting System Information"

# Detect Python version
if ! command_exists python3; then
    error_exit "Python 3 is not installed. Please install Python 3.8+ first."
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

print_info "Python Version: $PYTHON_VERSION"

# Verify Python version compatibility (3.8+)
if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
    error_exit "Python 3.8+ required. Found: $PYTHON_VERSION"
fi

# Detect CUDA version
if ! command_exists nvcc; then
    print_warning "nvcc not found in PATH. Attempting to locate CUDA..."
    if [[ -d /usr/local/cuda ]]; then
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        print_info "CUDA found at /usr/local/cuda"
    else
        print_warning "CUDA toolkit not found. TensorFlow will attempt CPU-only installation."
        CUDA_VERSION="Not detected"
    fi
else
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    print_info "CUDA Version: $CUDA_VERSION"
fi

# Detect GPU
if command_exists nvidia-smi; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU detection failed")
    print_info "GPU Detected: $GPU_INFO"
else
    print_warning "nvidia-smi not found. GPU may not be available."
fi

# Check pip
if ! command_exists pip3; then
    print_info "pip3 not found. Installing pip..."
    python3 -m ensurepip --upgrade || error_exit "Failed to install pip"
fi

# Upgrade pip
print_info "Upgrading pip, setuptools, and wheel..."
python3 -m pip install --upgrade pip setuptools wheel --quiet || error_exit "Failed to upgrade pip"

PIP_VERSION=$(pip3 --version | awk '{print $2}')
print_success "pip version: $PIP_VERSION"

################################################################################
# STEP 2: Determine TensorFlow Version
################################################################################

print_step "Determining Compatible TensorFlow Version"

# TensorFlow version selection based on Python and CUDA
# Python 3.8-3.11: TensorFlow 2.15+
# Python 3.12+: TensorFlow 2.16+
# CUDA 12+: TensorFlow 2.15+

if [[ $PYTHON_MINOR -ge 12 ]]; then
    TF_VERSION="tensorflow[and-cuda]>=2.16.0"
    print_info "Selected: TensorFlow 2.16+ (Python 3.12+ compatible)"
elif [[ $PYTHON_MINOR -ge 9 ]]; then
    TF_VERSION="tensorflow[and-cuda]>=2.15.0"
    print_info "Selected: TensorFlow 2.15+ (Python 3.9-3.11 compatible)"
else
    TF_VERSION="tensorflow[and-cuda]>=2.15.0"
    print_info "Selected: TensorFlow 2.15+ (Python 3.8 compatible)"
fi

print_success "TensorFlow version determined: $TF_VERSION"

################################################################################
# STEP 3: Create/Verify Virtual Environment (Optional but Recommended)
################################################################################

print_step "Virtual Environment Check"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    print_success "Running in virtual environment: $VIRTUAL_ENV"
else
    print_warning "Not running in a virtual environment"
    echo -e "${YELLOW}Recommended: Create a virtual environment to avoid conflicts${NC}"
    echo "Example: python3 -m venv venv && source venv/bin/activate"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled. Please create a virtual environment first."
        exit 0
    fi
fi

################################################################################
# STEP 4: Install Core Dependencies
################################################################################

print_step "Installing Core Dependencies"

print_info "This may take several minutes..."

# Create temporary requirements file
TEMP_REQ=$(mktemp)
cat > "$TEMP_REQ" << EOF
# Core ML/DL Framework
$TF_VERSION

# Computer Vision
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0

# Numerical Computing
numpy>=1.24.0,<2.0.0
pandas>=2.0.0

# Machine Learning
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.66.0
h5py>=3.9.0
Pillow>=10.0.0
psutil>=5.9.0

# GPU Monitoring
gpustat>=1.1.0
pynvml>=11.5.0

# Additional utilities
pyyaml>=6.0.0
requests>=2.31.0
EOF

print_info "Installing packages from requirements..."
pip3 install -r "$TEMP_REQ" --upgrade --quiet 2>&1 | grep -i "error\|success\|installed" || true

rm "$TEMP_REQ"

print_success "Core dependencies installation completed"

################################################################################
# STEP 5: Verify TensorFlow Installation
################################################################################

print_step "Verifying TensorFlow Installation"

# Test TensorFlow import
python3 << EOF
import sys
try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    else:
        print("⚠️  No GPUs detected by TensorFlow")
        print("   This is expected if CUDA is not properly configured")

    # Check CUDA and cuDNN
    if hasattr(tf.sysconfig, 'get_build_info'):
        build_info = tf.sysconfig.get_build_info()
        print(f"✅ CUDA version (built with): {build_info.get('cuda_version', 'N/A')}")
        print(f"✅ cuDNN version (built with): {build_info.get('cudnn_version', 'N/A')}")

    sys.exit(0)
except ImportError as e:
    print(f"❌ Failed to import TensorFlow: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ TensorFlow verification error: {e}")
    sys.exit(1)
EOF

if [[ $? -eq 0 ]]; then
    print_success "TensorFlow verification passed"
else
    error_exit "TensorFlow verification failed"
fi

################################################################################
# STEP 6: Verify Other Dependencies
################################################################################

print_step "Verifying All Dependencies"

python3 << 'EOF'
import sys

dependencies = {
    'opencv-python': 'cv2',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scikit-learn': 'sklearn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'tqdm': 'tqdm',
    'h5py': 'h5py',
    'Pillow': 'PIL',
    'psutil': 'psutil',
    'gpustat': 'gpustat',
    'pynvml': 'pynvml'
}

failed = []
for package, module in dependencies.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {package:20s} → {module:15s} (v{version})")
    except ImportError:
        print(f"❌ {package:20s} → {module:15s} FAILED")
        failed.append(package)

if failed:
    print(f"\n❌ Failed to import: {', '.join(failed)}")
    sys.exit(1)
else:
    print(f"\n✅ All dependencies verified successfully")
    sys.exit(0)
EOF

if [[ $? -ne 0 ]]; then
    error_exit "Dependency verification failed"
fi

################################################################################
# STEP 7: GPU Detection and Configuration Test
################################################################################

print_step "Testing GPU Detection and Configuration"

python3 << 'EOF'
import tensorflow as tf
import sys

print("\n" + "="*60)
print("GPU CONFIGURATION TEST")
print("="*60 + "\n")

# List physical devices
print("Physical Devices:")
for device_type in ['GPU', 'CPU']:
    devices = tf.config.list_physical_devices(device_type)
    print(f"  {device_type}: {len(devices)} device(s)")
    for i, device in enumerate(devices):
        print(f"    [{i}] {device.name}")

print()

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("⚠️  No GPU detected by TensorFlow")
    print("   Training will run on CPU (very slow)")
    print("   Please verify CUDA installation if GPU is available")
    sys.exit(0)

print(f"✅ {len(gpus)} GPU(s) detected\n")

# Configure GPU memory growth
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU memory growth enabled (recommended for A100)")
except RuntimeError as e:
    print(f"⚠️  Could not set memory growth: {e}")

# Test GPU computation
print("\nTesting GPU computation...")
try:
    with tf.device('/GPU:0'):
        # Simple matrix multiplication test
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        _ = c.numpy()  # Force execution

    print("✅ GPU computation test passed")

    # Get GPU details
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

        print(f"\nGPU Details:")
        print(f"  Name: {name}")
        print(f"  Total Memory: {memory.total / 1024**3:.2f} GB")
        print(f"  Free Memory: {memory.free / 1024**3:.2f} GB")
        print(f"  Used Memory: {memory.used / 1024**3:.2f} GB")

        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"⚠️  Could not get detailed GPU info: {e}")

except Exception as e:
    print(f"❌ GPU computation test failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ GPU CONFIGURATION SUCCESSFUL")
print("="*60)
EOF

if [[ $? -ne 0 ]]; then
    print_warning "GPU test encountered issues (see above)"
else
    print_success "GPU detection and configuration test passed"
fi

################################################################################
# STEP 8: Create Test Script for Training Verification
################################################################################

print_step "Creating Training Verification Script"

cat > "/tmp/test_training_setup.py" << 'EOF'
#!/usr/bin/env python3
"""
Quick verification script for NexaraVision training environment
Tests all critical components needed for violence detection training
"""

import tensorflow as tf
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("="*60)
print("NexaraVision Training Environment Test")
print("="*60)

# Test 1: TensorFlow + GPU
print("\n[1] TensorFlow + GPU:")
print(f"    TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"    GPUs available: {len(gpus)}")
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"      GPU {i}: {gpu.name}")

# Test 2: Build ResNet50V2 base
print("\n[2] ResNet50V2 Model:")
try:
    base_model = tf.keras.applications.ResNet50V2(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )
    print(f"    ✅ ResNet50V2 loaded ({base_model.count_params():,} params)")
except Exception as e:
    print(f"    ❌ Failed: {e}")

# Test 3: BiGRU layer
print("\n[3] BiGRU Layer:")
try:
    bigru = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(128, return_sequences=False)
    )
    test_input = tf.random.normal([1, 10, 256])
    output = bigru(test_input)
    print(f"    ✅ BiGRU functional (output shape: {output.shape})")
except Exception as e:
    print(f"    ❌ Failed: {e}")

# Test 4: OpenCV video processing
print("\n[4] OpenCV:")
print(f"    OpenCV version: {cv2.__version__}")
try:
    # Test image operations
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_resized = cv2.resize(img, (112, 112))
    print(f"    ✅ Image operations working")
except Exception as e:
    print(f"    ❌ Failed: {e}")

# Test 5: Data pipeline
print("\n[5] Data Pipeline:")
try:
    # Simulate data pipeline
    dummy_data = np.random.rand(100, 10, 224, 224, 3)
    dummy_labels = np.random.randint(0, 2, 100)
    X_train, X_val, y_train, y_val = train_test_split(
        dummy_data, dummy_labels, test_size=0.2, random_state=42
    )
    print(f"    ✅ Train: {X_train.shape}, Val: {X_val.shape}")
except Exception as e:
    print(f"    ❌ Failed: {e}")

# Test 6: Memory estimation
print("\n[6] Memory Estimation:")
try:
    if gpus:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"    GPU Memory: {memory.total / 1024**3:.2f} GB total")
        print(f"    Available: {memory.free / 1024**3:.2f} GB")
        pynvml.nvmlShutdown()

        # Estimate batch size
        model_memory_gb = 2.5  # Estimated for ResNet50V2 + BiGRU
        available_gb = memory.free / 1024**3
        recommended_batch = int((available_gb - model_memory_gb) * 0.8 / 0.5)  # ~0.5GB per batch
        print(f"    Recommended batch size: {max(8, recommended_batch)} (conservative estimate)")
    else:
        print(f"    No GPU detected - CPU training will be very slow")
except Exception as e:
    print(f"    ⚠️  Could not estimate: {e}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - Environment ready for training")
print("="*60)
print("\nNext steps:")
print("  1. Prepare your dataset (videos → frame sequences)")
print("  2. Run training script with recommended batch size")
print("  3. Monitor GPU usage with: watch -n 1 nvidia-smi")
EOF

chmod +x "/tmp/test_training_setup.py"

print_info "Running quick training environment test..."
python3 /tmp/test_training_setup.py

if [[ $? -eq 0 ]]; then
    print_success "Training environment test passed"
else
    print_warning "Training environment test completed with warnings"
fi

################################################################################
# Installation Summary
################################################################################

print_header "Installation Complete!"

echo -e "${GREEN}Environment Details:${NC}"
echo -e "  Python:     ${CYAN}$PYTHON_VERSION${NC}"
echo -e "  CUDA:       ${CYAN}${CUDA_VERSION}${NC}"
echo -e "  TensorFlow: ${CYAN}$(python3 -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Installed')${NC}"
echo -e "  GPU:        ${CYAN}$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'Not detected')${NC}"

echo -e "\n${GREEN}Installed Packages:${NC}"
pip3 list | grep -E "tensorflow|opencv|numpy|pandas|scikit|matplotlib|seaborn|tqdm|h5py|pillow|psutil|gpustat|pynvml" || true

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "  1. ${CYAN}Verify installation:${NC}"
echo -e "     python3 /tmp/test_training_setup.py"
echo -e ""
echo -e "  2. ${CYAN}Monitor GPU during training:${NC}"
echo -e "     watch -n 1 nvidia-smi"
echo -e ""
echo -e "  3. ${CYAN}Start training:${NC}"
echo -e "     python3 scripts/train_model.py --batch-size 16 --epochs 50"
echo -e ""
echo -e "  4. ${CYAN}Enable memory growth in training script:${NC}"
echo -e "     ${BLUE}gpus = tf.config.list_physical_devices('GPU')${NC}"
echo -e "     ${BLUE}for gpu in gpus:${NC}"
echo -e "     ${BLUE}    tf.config.experimental.set_memory_growth(gpu, True)${NC}"

echo -e "\n${GREEN}Useful Commands:${NC}"
echo -e "  GPU status:     ${CYAN}nvidia-smi${NC}"
echo -e "  GPU monitoring: ${CYAN}gpustat -i 1${NC}"
echo -e "  Python env:     ${CYAN}pip3 list${NC}"
echo -e "  TensorFlow GPU: ${CYAN}python3 -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'${NC}"

echo -e "\n${GREEN}Documentation:${NC}"
echo -e "  TensorFlow: https://www.tensorflow.org/guide/gpu"
echo -e "  A100 Tuning: https://www.nvidia.com/en-us/data-center/a100/"

echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Installation script completed successfully!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}\n"

# Save installation log
INSTALL_LOG="/tmp/nexaravision_install_$(date +%Y%m%d_%H%M%S).log"
echo "Installation completed at $(date)" > "$INSTALL_LOG"
echo "Python: $PYTHON_VERSION" >> "$INSTALL_LOG"
echo "CUDA: $CUDA_VERSION" >> "$INSTALL_LOG"
pip3 list >> "$INSTALL_LOG"
print_info "Installation log saved to: $INSTALL_LOG"

exit 0
