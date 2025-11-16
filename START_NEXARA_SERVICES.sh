#!/bin/bash

# NexaraVision Services Startup Script
# Starts ML Service (8003), Backend (8002), and Frontend (8001)

set -e

PROJECT_DIR="/home/admin/Desktop/NexaraVision"
LOG_DIR="$PROJECT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create logs directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   NexaraVision Services Startup${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to check if port is in use
check_port() {
    if lsof -i :$1 > /dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill process on port
kill_port() {
    if check_port $1; then
        echo -e "${YELLOW}Killing existing process on port $1...${NC}"
        fuser -k $1/tcp 2>/dev/null || true
        sleep 1
    fi
}

# Stop all services
stop_services() {
    echo -e "\n${RED}Stopping all services...${NC}"
    kill_port 8001  # Frontend
    kill_port 8002  # Backend
    kill_port 8003  # ML Service
    echo -e "${GREEN}All services stopped.${NC}"
    exit 0
}

# Trap SIGINT (Ctrl+C) to stop all services
trap stop_services SIGINT

# Check for required dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: node not found${NC}"
    exit 1
fi

echo -e "${GREEN}Dependencies OK${NC}"

# Kill any existing services on these ports
echo -e "\n${YELLOW}Cleaning up existing processes...${NC}"
kill_port 8001
kill_port 8002
kill_port 8003

# 1. Start ML Service (FastAPI on port 8003)
echo -e "\n${BLUE}[1/3] Starting ML Service on port 8003...${NC}"
cd "$PROJECT_DIR/ml_service"

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt --quiet 2>/dev/null || pip install -r requirements.txt

echo -e "${GREEN}Starting ML Service...${NC}"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8003 --log-level info > "$LOG_DIR/ml_service.log" 2>&1 &
ML_PID=$!
echo -e "${GREEN}ML Service PID: $ML_PID${NC}"

# Wait for ML service to be ready
echo -e "${YELLOW}Waiting for ML Service to load model (this may take 30-60 seconds)...${NC}"
for i in {1..60}; do
    if curl -s http://localhost:8003/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}ML Service is ready!${NC}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo -e "${RED}ML Service failed to start. Check logs: $LOG_DIR/ml_service.log${NC}"
        tail -20 "$LOG_DIR/ml_service.log"
        exit 1
    fi
    sleep 1
    echo -n "."
done

deactivate

# 2. Start Backend Service (NestJS on port 8002)
echo -e "\n${BLUE}[2/3] Starting Backend Service on port 8002...${NC}"
cd "$PROJECT_DIR/web_app_backend"

if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    npm install
fi

echo -e "${GREEN}Starting Backend Service...${NC}"
npm run start:dev > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}Backend Service PID: $BACKEND_PID${NC}"

# Wait for backend to be ready
echo -e "${YELLOW}Waiting for Backend Service...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8002/api > /dev/null 2>&1; then
        echo -e "${GREEN}Backend Service is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Backend Service failed to start. Check logs: $LOG_DIR/backend.log${NC}"
        tail -20 "$LOG_DIR/backend.log"
        exit 1
    fi
    sleep 1
    echo -n "."
done

# 3. Start Frontend Service (Next.js on port 8001)
echo -e "\n${BLUE}[3/3] Starting Frontend Service on port 8001...${NC}"
cd "$PROJECT_DIR/web_app_nextjs"

if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

echo -e "${GREEN}Starting Frontend Service...${NC}"
PORT=8001 npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}Frontend Service PID: $FRONTEND_PID${NC}"

# Wait for frontend to be ready
echo -e "${YELLOW}Waiting for Frontend Service...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8001 > /dev/null 2>&1; then
        echo -e "${GREEN}Frontend Service is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Frontend Service failed to start. Check logs: $LOG_DIR/frontend.log${NC}"
        tail -20 "$LOG_DIR/frontend.log"
        exit 1
    fi
    sleep 1
    echo -n "."
done

# Print summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}   All Services Started Successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "  ${BLUE}Frontend:${NC}    http://localhost:8001"
echo -e "  ${BLUE}Backend:${NC}     http://localhost:8002"
echo -e "  ${BLUE}ML Service:${NC}  http://localhost:8003"
echo -e ""
echo -e "  ${BLUE}Live Detection:${NC}  http://localhost:8001/live"
echo -e "  ${BLUE}API Docs:${NC}        http://localhost:8003/docs"
echo -e ""
echo -e "  ${YELLOW}Logs:${NC}"
echo -e "    ML Service: $LOG_DIR/ml_service.log"
echo -e "    Backend:    $LOG_DIR/backend.log"
echo -e "    Frontend:   $LOG_DIR/frontend.log"
echo -e ""
echo -e "  ${RED}Press Ctrl+C to stop all services${NC}"
echo -e ""

# Keep script running and monitor services
while true; do
    # Check if all services are still running
    if ! kill -0 $ML_PID 2>/dev/null; then
        echo -e "${RED}ML Service stopped unexpectedly!${NC}"
        stop_services
    fi
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo -e "${RED}Backend Service stopped unexpectedly!${NC}"
        stop_services
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo -e "${RED}Frontend Service stopped unexpectedly!${NC}"
        stop_services
    fi
    sleep 5
done
