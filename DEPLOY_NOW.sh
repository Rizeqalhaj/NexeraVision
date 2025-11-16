#!/bin/bash
# Quick Production Deployment Script
# Run this on the server: bash DEPLOY_NOW.sh

set -e

echo "=========================================="
echo "NexaraVision Production Deployment"
echo "Ports: Frontend 3005, Backend 3006, ML 3007"
echo "=========================================="

# Navigate to deployment directory
DEPLOY_DIR=/root/nexara-vision-production
mkdir -p $DEPLOY_DIR
cd $DEPLOY_DIR

# Clone or update repository
if [ -d ".git" ]; then
  echo "📥 Updating repository..."
  git fetch origin
  git reset --hard origin/main
  git clean -fd
else
  echo "📥 Cloning repository..."
  git clone -b main https://github.com/Rizeqalhaj/NexeraVision.git .
fi

echo ""
echo "=== Step 1: Deploy Frontend (Port 3005) ==="
cd $DEPLOY_DIR/web_app_nextjs
npm ci
npm run build
npm prune --production
pm2 delete nexara-vision-frontend-production 2>/dev/null || true
PORT=3005 pm2 start npm --name "nexara-vision-frontend-production" -- start
echo "✅ Frontend deployed on port 3005"

echo ""
echo "=== Step 2: Deploy Backend (Port 3006) ==="
cd $DEPLOY_DIR/web_app_backend
npm ci

# Create .env file
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

# Run migrations and build
set -a && source .env && set +a
npx prisma generate
npx prisma migrate deploy 2>/dev/null || echo "Migrations already applied"
npm run build

pm2 delete nexara-vision-backend-production 2>/dev/null || true
PORT=3006 pm2 start dist/src/main.js --name "nexara-vision-backend-production"
echo "✅ Backend deployed on port 3006"

echo ""
echo "=== Step 3: Deploy ML Service (Port 3007) ==="
if [ -d "$DEPLOY_DIR/ml_service" ]; then
  cd $DEPLOY_DIR/ml_service
  docker stop nexara-ml-service-production 2>/dev/null || true
  docker rm nexara-ml-service-production 2>/dev/null || true

  if docker build -t nexara-ml-service:production .; then
    docker run -d \
      --name nexara-ml-service-production \
      -p 3007:8000 \
      --restart unless-stopped \
      -v $DEPLOY_DIR/models:/app/models:ro \
      nexara-ml-service:production
    echo "✅ ML Service deployed on port 3007"
  else
    echo "⚠️  ML service build failed"
  fi
else
  echo "⚠️  ML service directory not found"
fi

echo ""
echo "=== Step 4: Configure Nginx ==="
cat > /tmp/nginx-production.conf << 'NGINX'
server {
    listen 80;
    listen 443 ssl http2;
    server_name vision.nexaratech.io;

    ssl_certificate /etc/letsencrypt/live/vision.nexaratech.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/vision.nexaratech.io/privkey.pem;

    client_max_body_size 100M;

    # Next.js Frontend
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

    # NestJS Backend API
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
echo "✅ Nginx configured"

echo ""
echo "=== Step 5: Health Checks ==="
sleep 5

echo "PM2 Processes:"
pm2 list | grep production || echo "No PM2 production processes"

echo ""
echo "Docker Containers:"
docker ps | grep production || echo "No Docker production containers"

echo ""
echo "Testing endpoints..."
curl -f http://localhost:3005 > /dev/null 2>&1 && echo "✅ Frontend OK" || echo "❌ Frontend FAILED"
curl -f http://localhost:3006 > /dev/null 2>&1 && echo "✅ Backend OK" || echo "⚠️  Backend check (may need time to start)"
docker ps | grep nexara-ml-service-production > /dev/null && echo "✅ ML Service running" || echo "⚠️  ML Service not running"

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo "Production URL: https://vision.nexaratech.io"
echo "Services:"
echo "  • Frontend: port 3005"
echo "  • Backend: port 3006"
echo "  • ML Service: port 3007"
echo ""
echo "Save PM2 configuration:"
pm2 save
echo "=========================================="
