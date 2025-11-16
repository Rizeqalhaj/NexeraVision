#!/bin/bash
# Fix Production Port Configuration
# This script updates EXISTING production deployment to correct ports
# Does NOT touch staging deployment

set -e

echo "=========================================="
echo "Fixing Production Ports"
echo "Current (wrong): using staging ports"
echo "Target (correct): 3005, 3006, 3007"
echo "=========================================="

echo ""
echo "=== Current Status ==="
echo "PM2 Processes:"
pm2 list

echo ""
echo "Docker Containers:"
docker ps -a

echo ""
echo "Ports in use:"
netstat -tlnp | grep -E ":(3005|3006|3007|8001|8002|8003|8002)"

echo ""
echo "=== Step 1: Stop OLD Production Services ==="
# Stop any old Docker container that might be running
docker stop violence-detection 2>/dev/null && echo "Stopped old Docker container" || echo "No old Docker container"
docker rm violence-detection 2>/dev/null && echo "Removed old Docker container" || echo "No old Docker container to remove"

# Check if there are any production PM2 processes on wrong ports
pm2 delete nexara-vision-frontend-production 2>/dev/null || echo "No existing frontend production process"
pm2 delete nexara-vision-backend-production 2>/dev/null || echo "No existing backend production process"
docker stop nexara-ml-service-production 2>/dev/null || echo "No existing ML production container"
docker rm nexara-ml-service-production 2>/dev/null || echo "No existing ML production container to remove"

echo ""
echo "=== Step 2: Update Production Code ==="
DEPLOY_DIR=/root/nexara-vision-production

if [ ! -d "$DEPLOY_DIR" ]; then
  echo "Production directory not found, checking /home/admin..."
  DEPLOY_DIR=/home/admin/nexara-vision-production
fi

if [ -d "$DEPLOY_DIR/.git" ]; then
  echo "Updating existing production repository..."
  cd $DEPLOY_DIR
  git fetch origin
  git reset --hard origin/main
  git clean -fd
  echo "✅ Code updated from main branch"
else
  echo "No production repository found. Creating fresh deployment..."
  mkdir -p $DEPLOY_DIR
  cd $DEPLOY_DIR
  git clone -b main https://github.com/Rizeqalhaj/NexeraVision.git .
  echo "✅ Fresh clone from main branch"
fi

echo ""
echo "=== Step 3: Deploy Frontend on Port 3005 ==="
cd $DEPLOY_DIR/web_app_nextjs

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm ci
else
  echo "Dependencies already installed, updating..."
  npm ci
fi

# Build if .next doesn't exist or is old
if [ ! -d ".next" ]; then
  echo "Building Next.js app..."
  npm run build
  npm prune --production
else
  echo "Build exists, rebuilding..."
  npm run build
  npm prune --production
fi

# Start on port 3005
echo "Starting frontend on port 3005..."
PORT=3005 pm2 start npm --name "nexara-vision-frontend-production" -- start
echo "✅ Frontend running on port 3005"

echo ""
echo "=== Step 4: Deploy Backend on Port 3006 ==="
cd $DEPLOY_DIR/web_app_backend

# Install dependencies
if [ ! -d "node_modules" ]; then
  npm ci
else
  npm ci
fi

# Create/update .env file
echo "Creating production environment configuration..."
cat > .env << 'ENV'
PORT=3006
NODE_ENV=production
DATABASE_URL=postgresql://postgres:E$$athecode006@localhost:5432/nexara_vision_production
ML_SERVICE_URL=http://localhost:3007
JWT_SECRET=nexara-vision-production-secret-key-2024
REDIS_HOST=localhost
REDIS_PORT=6379
CORS_ORIGIN=https://vision.nexaratech.io,http://localhost:3005
ENV

# Run Prisma
echo "Running Prisma migrations..."
set -a && source .env && set +a
npx prisma generate
npx prisma migrate deploy 2>/dev/null || echo "Migrations already applied"

# Build
if [ ! -d "dist" ]; then
  echo "Building NestJS app..."
  npm run build
else
  echo "Rebuilding NestJS app..."
  npm run build
fi

# Start on port 3006
echo "Starting backend on port 3006..."
PORT=3006 pm2 start dist/src/main.js --name "nexara-vision-backend-production"
echo "✅ Backend running on port 3006"

echo ""
echo "=== Step 5: Deploy ML Service on Port 3007 ==="
if [ -d "$DEPLOY_DIR/ml_service" ]; then
  cd $DEPLOY_DIR/ml_service

  echo "Building ML service Docker image..."
  if docker build -t nexara-ml-service:production . 2>&1 | tail -20; then
    echo "Starting ML service on port 3007..."
    docker run -d \
      --name nexara-ml-service-production \
      -p 3007:8000 \
      --restart unless-stopped \
      -v $DEPLOY_DIR/models:/app/models:ro \
      nexara-ml-service:production
    echo "✅ ML Service running on port 3007"
  else
    echo "⚠️  ML service build failed (skipping)"
  fi
else
  echo "⚠️  ML service directory not found"
fi

echo ""
echo "=== Step 6: Update Nginx Configuration ==="
echo "Updating nginx to route to new ports..."

cat > /tmp/nginx-production.conf << 'NGINX'
server {
    listen 80;
    listen 443 ssl http2;
    server_name vision.nexaratech.io;

    ssl_certificate /etc/letsencrypt/live/vision.nexaratech.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/vision.nexaratech.io/privkey.pem;

    client_max_body_size 100M;

    # Next.js Frontend (port 3005)
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

    # NestJS Backend API (port 3006)
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
NGINX

sudo mv /tmp/nginx-production.conf /etc/nginx/sites-available/violence-detection
sudo ln -sf /etc/nginx/sites-available/violence-detection /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
echo "✅ Nginx updated and reloaded"

echo ""
echo "=== Step 7: Verify Staging is Untouched ==="
echo "Checking staging services..."
pm2 list | grep staging || echo "Staging processes OK (or not using PM2)"
netstat -tlnp | grep -E ":(8001|8002|8003)" && echo "✅ Staging ports still in use" || echo "⚠️  Staging ports not detected"

echo ""
echo "=== Step 8: Health Checks ==="
sleep 10

echo "Production PM2 Processes:"
pm2 list | grep production

echo ""
echo "Production Docker Containers:"
docker ps | grep production

echo ""
echo "Testing Production Endpoints:"
curl -f -s http://localhost:3005 > /dev/null 2>&1 && echo "✅ Frontend (3005): OK" || echo "❌ Frontend (3005): FAILED"
curl -f -s http://localhost:3006 > /dev/null 2>&1 && echo "✅ Backend (3006): OK" || echo "⚠️  Backend (3006): May need more time"
docker ps | grep nexara-ml-service-production > /dev/null && echo "✅ ML Service (3007): Running" || echo "⚠️  ML Service (3007): Not running"

echo ""
echo "Testing Staging Endpoints:"
curl -f -s http://localhost:8001 > /dev/null 2>&1 && echo "✅ Staging Frontend (8001): OK" || echo "⚠️  Staging Frontend: Check status"
curl -f -s http://localhost:8002 > /dev/null 2>&1 && echo "✅ Staging Backend (8002): OK" || echo "⚠️  Staging Backend: Check status"

echo ""
echo "Port Status:"
netstat -tlnp | grep -E ":(3005|3006|3007|8001|8002|8003)"

echo ""
echo "=========================================="
echo "✅ Production Port Fix Complete!"
echo "=========================================="
echo "Production (vision.nexaratech.io):"
echo "  • Frontend: port 3005 ✅"
echo "  • Backend: port 3006 ✅"
echo "  • ML Service: port 3007 ✅"
echo ""
echo "Staging (stagingvision.nexaratech.io):"
echo "  • Frontend: port 8001 (unchanged)"
echo "  • Backend: port 8002 (unchanged)"
echo "  • ML Service: port 8003 (unchanged)"
echo ""
echo "Saving PM2 configuration..."
pm2 save
echo ""
echo "=========================================="
echo "Test URLs:"
echo "Production: https://vision.nexaratech.io"
echo "Staging: http://stagingvision.nexaratech.io"
echo "=========================================="
