# Fix "Network error occurred" - Start ML Service

## Problem
The frontend can't reach `/api/upload` because the ML service isn't running.

## Solution: Start the ML Service

### Option 1: Start with Docker (Recommended)
```bash
cd /home/admin/Desktop/NexaraVision/ml_service

# Start Docker Desktop first, then:
docker build -t nexara-ml-service:latest .

docker run -d \
  --name nexara-ml-service \
  -p 8003:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_PATH=/app/models/initial_best_model.keras \
  --restart unless-stopped \
  nexara-ml-service:latest

# Check if it's running:
docker logs nexara-ml-service

# Test it:
curl http://localhost:8003/
```

### Option 2: Run ML Service Directly (Python)
```bash
cd /home/admin/Desktop/NexaraVision/ml_service

# Install dependencies first (if not done):
pip3 install -r requirements.txt

# Run the service:
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8003

# Service will start on http://localhost:8003
```

### Option 3: Update Backend to Proxy to Remote ML Service
If you want to use the production ML service temporarily:
```bash
# Edit web_app_nextjs/.env.local (create if doesn't exist)
echo "NEXT_PUBLIC_ML_API_URL=https://vision.nexaratech.io" > web_app_nextjs/.env.local

# Restart frontend:
cd web_app_nextjs
npm run dev
```

## After Starting ML Service

The frontend (localhost:8001) needs to know where to find it. Check:

```bash
# Does backend proxy to ML service?
# OR
# Does frontend call ML service directly?
```

Let me know which option you want to try!
