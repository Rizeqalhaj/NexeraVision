#!/bin/bash
# NexaraVision - Start All Services on Ports 8001-8004

echo "=========================================="
echo "NexaraVision - Starting All Services"
echo "=========================================="

# Port 8001: Next.js Frontend
echo ""
echo "🌐 Starting Frontend (Port 8001)..."
cd /home/admin/Desktop/NexaraVision/web_app_nextjs
npm run dev -- -p 8001 > /tmp/frontend-8001.log 2>&1 &
FRONTEND_PID=$!
echo "   PID: $FRONTEND_PID"
sleep 2
curl -s http://localhost:8001 > /dev/null && echo "   ✅ Frontend running" || echo "   ⏳ Frontend starting..."

# Port 8002: NestJS Backend (requires database)
echo ""
echo "🔧 Backend (Port 8002) - Requires database configuration"
echo "   Skipping... (configure database in web_app_backend/.env first)"

# Port 8003: ML Service (requires trained model)
echo ""
echo "🤖 ML Service (Port 8003) - Requires trained model"
echo "   Skipping... (place model at ml_service/models/ultimate_best_model.h5 first)"

# Port 8004: GridDetector Backend
echo ""
echo "🔍 Starting GridDetector (Port 8004)..."
cd /home/admin/Desktop/NexaraVision/grid_detection

# Check if dependencies are installed
if ! source venv/bin/activate 2>/dev/null || ! python3 -c "import cv2" 2>/dev/null; then
    echo "   ⏳ Installing dependencies (this may take 5-10 minutes)..."
    python3 -m venv venv 2>/dev/null || true
    source venv/bin/activate
    pip install -q fastapi uvicorn numpy opencv-python
fi

source venv/bin/activate
python3 -m uvicorn api_integration:app --host 0.0.0.0 --port 8004 --reload > /tmp/griddetector-8004.log 2>&1 &
GRID_PID=$!
echo "   PID: $GRID_PID"
sleep 2
curl -s http://localhost:8004/health > /dev/null && echo "   ✅ GridDetector running" || echo "   ⏳ GridDetector starting..."

echo ""
echo "=========================================="
echo "Service Status Summary"
echo "=========================================="
curl -s http://localhost:8001 > /dev/null && echo "✅ Port 8001: Frontend" || echo "❌ Port 8001: Frontend"
curl -s http://localhost:8002 > /dev/null && echo "✅ Port 8002: Backend" || echo "❌ Port 8002: Backend"
curl -s http://localhost:8003 > /dev/null && echo "✅ Port 8003: ML Service" || echo "❌ Port 8003: ML Service"
curl -s http://localhost:8004/health > /dev/null && echo "✅ Port 8004: GridDetector" || echo "❌ Port 8004: GridDetector"

echo ""
echo "=========================================="
echo "Access URLs"
echo "=========================================="
echo "Frontend:      http://localhost:8001"
echo "Frontend Live: http://localhost:8001/live"
echo "GridDetector:  http://localhost:8004/docs"
echo ""
echo "Logs:"
echo "  Frontend:     tail -f /tmp/frontend-8001.log"
echo "  GridDetector: tail -f /tmp/griddetector-8004.log"
echo ""
echo "=========================================="
