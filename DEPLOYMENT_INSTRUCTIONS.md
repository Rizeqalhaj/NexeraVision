# Quick Fix for Production Ports

## Problem
Production is using wrong ports (staging ports). Need to fix to use:
- Frontend: 3005 (instead of staging's 8001)
- Backend: 3006 (instead of staging's 8002)
- ML Service: 3007 (instead of staging's 8003)

## Solution: Run Fix Script on Server

### Option 1: Copy and Run Script

```bash
# 1. Copy the fix script to the server
scp FIX_PRODUCTION_PORTS.sh root@31.57.166.18:/root/

# 2. SSH to the server
ssh root@31.57.166.18
# Password: qMRF2Y5Z44fBP1kANKcJHX61

# 3. Run the fix script
bash /root/FIX_PRODUCTION_PORTS.sh

# 4. Verify both environments
curl https://vision.nexaratech.io
curl http://stagingvision.nexaratech.io
```

### Option 2: Direct Remote Execution

```bash
# Run the script directly via SSH
cat FIX_PRODUCTION_PORTS.sh | ssh root@31.57.166.18 'bash -s'
# Password: qMRF2Y5Z44fBP1kANKcJHX61
```

### Option 3: Manual Commands on Server

If you're already logged into the server, run these commands:

```bash
# Stop old production services
pm2 delete nexara-vision-frontend-production 2>/dev/null || true
pm2 delete nexara-vision-backend-production 2>/dev/null || true
docker stop nexara-ml-service-production 2>/dev/null || true
docker rm nexara-ml-service-production 2>/dev/null || true

# Update production code
cd /root/nexara-vision-production || cd /home/admin/nexara-vision-production
git fetch origin && git reset --hard origin/main

# Deploy frontend on port 3005
cd web_app_nextjs
npm ci && npm run build && npm prune --production
PORT=3005 pm2 start npm --name "nexara-vision-frontend-production" -- start

# Deploy backend on port 3006
cd ../web_app_backend
npm ci
cat > .env << 'EOF'
PORT=3006
NODE_ENV=production
DATABASE_URL=postgresql://postgres:E$$athecode006@localhost:5432/nexara_vision_production
ML_SERVICE_URL=http://localhost:3007
JWT_SECRET=nexara-vision-production-secret-key-2024
REDIS_HOST=localhost
REDIS_PORT=6379
CORS_ORIGIN=https://vision.nexaratech.io,http://localhost:3005
EOF
source .env
npx prisma generate && npx prisma migrate deploy
npm run build
PORT=3006 pm2 start dist/src/main.js --name "nexara-vision-backend-production"

# Deploy ML service on port 3007
cd ../ml_service
docker build -t nexara-ml-service:production .
docker run -d --name nexara-ml-service-production -p 3007:8000 --restart unless-stopped nexara-ml-service:production

# Update nginx
cat > /etc/nginx/sites-available/violence-detection << 'EOF'
server {
    listen 80;
    listen 443 ssl http2;
    server_name vision.nexaratech.io;
    ssl_certificate /etc/letsencrypt/live/vision.nexaratech.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/vision.nexaratech.io/privkey.pem;
    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:3005;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api {
        proxy_pass http://localhost:3006;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;
        send_timeout 600;
    }
}
EOF

nginx -t && systemctl reload nginx
pm2 save
```

## Verification

After running the fix, verify:

```bash
# Check services
pm2 list
docker ps

# Check ports
netstat -tlnp | grep -E ":(3005|3006|3007|8001|8002|8003)"

# Test endpoints
curl http://localhost:3005  # Production frontend
curl http://localhost:3006  # Production backend
curl http://localhost:8001  # Staging frontend (should still work)
curl http://localhost:8002  # Staging backend (should still work)
```

## Expected Result

**Production (vision.nexaratech.io)**:
- ✅ Frontend on port 3005
- ✅ Backend on port 3006
- ✅ ML Service on port 3007

**Staging (stagingvision.nexaratech.io)**:
- ✅ Frontend on port 8001 (unchanged)
- ✅ Backend on port 8002 (unchanged)
- ✅ ML Service on port 8003 (unchanged)
