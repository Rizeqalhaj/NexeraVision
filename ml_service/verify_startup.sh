#!/bin/bash

# NexaraVision ML Service Startup Verification Script
# Verifies all fixes are working correctly

echo "╔══════════════════════════════════════════════════════════╗"
echo "║    NexaraVision ML Service Startup Verification         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Python available
echo "CHECK 1: Python Installation"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ PASS${NC} - $PYTHON_VERSION"
else
    echo -e "${RED}❌ FAIL${NC} - Python3 not found"
    exit 1
fi
echo ""

# Check 2: Required dependencies
echo "CHECK 2: Dependencies"
REQUIRED_DEPS=(
    "fastapi"
    "tensorflow"
    "pydantic_settings"
    "websockets"
    "opencv-python"
    "numpy"
    "pillow"
)

MISSING_DEPS=()
for dep in "${REQUIRED_DEPS[@]}"; do
    if python3 -c "import ${dep//-/_}" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} $dep"
    else
        echo -e "${RED}❌${NC} $dep (missing)"
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Missing dependencies detected${NC}"
    echo "To install: pip install ${MISSING_DEPS[@]}"
    echo ""
fi
echo ""

# Check 3: Model file exists
echo "CHECK 3: Model File"
MODEL_PATHS=(
    "ml_service/models/best_model.h5"
    "models/best_model.h5"
    "downloaded_models/ultimate_best_model.h5"
    "downloaded_models/best_model.h5"
)

MODEL_FOUND=false
for path in "${MODEL_PATHS[@]}"; do
    if [ -f "$path" ]; then
        SIZE=$(du -h "$path" | cut -f1)
        echo -e "${GREEN}✅ FOUND${NC} - $path ($SIZE)"
        MODEL_FOUND=true
        break
    fi
done

if [ "$MODEL_FOUND" = false ]; then
    echo -e "${YELLOW}⚠️  No model file found in standard paths${NC}"
    echo "Set MODEL_PATH environment variable to custom location"
fi
echo ""

# Check 4: GPU availability (optional)
echo "CHECK 4: GPU Availability (Optional)"
if python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); exit(0 if gpus else 1)" 2>/dev/null; then
    GPU_COUNT=$(python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))")
    echo -e "${GREEN}✅ GPU DETECTED${NC} - $GPU_COUNT GPU(s) available"
    echo "   Service will use GPU acceleration"
else
    echo -e "${YELLOW}⚠️  No GPU detected${NC}"
    echo "   Service will run on CPU (slower but functional)"
fi
echo ""

# Check 5: Syntax validation
echo "CHECK 5: Code Syntax Validation"
FILES=(
    "app/core/gpu.py"
    "app/core/config.py"
    "app/models/violence_detector.py"
    "app/api/websocket.py"
    "app/main.py"
)

SYNTAX_ERRORS=0
for file in "${FILES[@]}"; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} $file"
    else
        echo -e "${RED}❌${NC} $file (syntax error)"
        SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
    fi
done

if [ $SYNTAX_ERRORS -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Syntax errors detected${NC}"
    exit 1
fi
echo ""

# Check 6: Port availability
echo "CHECK 6: Port Availability"
if ! lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Port 8000 available${NC}"
else
    echo -e "${YELLOW}⚠️  Port 8000 in use${NC}"
    echo "   Stop existing service or use different port"
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  Verification Summary                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if [ ${#MISSING_DEPS[@]} -eq 0 ] && [ $SYNTAX_ERRORS -eq 0 ] && [ "$MODEL_FOUND" = true ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED${NC}"
    echo ""
    echo "Service is ready to start!"
    echo ""
    echo "To start the service:"
    echo "  cd /home/admin/Desktop/NexaraVision/ml_service"
    echo "  python3 -m app.main"
    echo ""
    echo "Service will be available at:"
    echo "  - HTTP: http://localhost:8000"
    echo "  - WebSocket: ws://localhost:8000/api/ws/live"
    echo "  - Docs: http://localhost:8000/docs"
else
    echo -e "${YELLOW}⚠️  WARNINGS DETECTED${NC}"
    echo ""
    if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
        echo "Missing dependencies: ${MISSING_DEPS[@]}"
        echo "Install with: pip install -r requirements.txt"
        echo ""
    fi
    if [ "$MODEL_FOUND" = false ]; then
        echo "Model file not found - set MODEL_PATH environment variable"
        echo ""
    fi
fi

echo ""
echo "For detailed documentation, see:"
echo "  /home/admin/Desktop/NexaraVision/ML_SERVICE_FIX_SUMMARY.md"
echo ""
