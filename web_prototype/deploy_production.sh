#!/bin/bash
# Nexara Vision Enterprise - Production Deployment Script
# Deploys to production server with proper model mounting

set -e

echo "========================================="
echo "Nexara Vision Enterprise Deployment"
echo "========================================="

# Configuration
REMOTE_HOST="31.57.166.18"
REMOTE_USER="root"
REMOTE_PASSWORD="qMRF2Y5Z44fBP1kANKcJHX61"
CONTAINER_NAME="nexara-vision-detection"
IMAGE_NAME="nexara-vision:production-v2"
BACKEND_PORT=8005
MODELS_PATH="/home/admin/Desktop/NexaraVision/downloaded_models"

echo ""
echo "Step 1: Building production Docker image..."
docker build -f Dockerfile.production -t ${IMAGE_NAME} .

echo ""
echo "Step 2: Saving Docker image to tar..."
docker save ${IMAGE_NAME} | gzip > /tmp/nexara_vision_prod.tar.gz

echo ""
echo "Step 3: Copying image to production server..."
sshpass -p "${REMOTE_PASSWORD}" scp /tmp/nexara_vision_prod.tar.gz ${REMOTE_USER}@${REMOTE_HOST}:/tmp/

echo ""
echo "Step 4: Loading image on production server..."
sshpass -p "${REMOTE_PASSWORD}" ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
docker load < /tmp/nexara_vision_prod.tar.gz
rm /tmp/nexara_vision_prod.tar.gz
EOF

echo ""
echo "Step 5: Creating models directory on server..."
sshpass -p "${REMOTE_PASSWORD}" ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
mkdir -p /root/nexara_models
chmod 755 /root/nexara_models
EOF

echo ""
echo "Step 6: Copying models to production server..."
cd ${MODELS_PATH}
for model in best_model.h5 ultimate_best_model.h5 ensemble_m1_best.h5 ensemble_m2_best.h5 ensemble_m3_best.h5; do
    if [ -f "$model" ]; then
        echo "  Copying $model..."
        sshpass -p "${REMOTE_PASSWORD}" scp "$model" ${REMOTE_USER}@${REMOTE_HOST}:/root/nexara_models/
    fi
done

echo ""
echo "Step 7: Stopping old container (if exists)..."
sshpass -p "${REMOTE_PASSWORD}" ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
docker stop nexara-vision-detection 2>/dev/null || true
docker rm nexara-vision-detection 2>/dev/null || true
EOF

echo ""
echo "Step 8: Starting new container..."
sshpass -p "${REMOTE_PASSWORD}" ssh ${REMOTE_USER}@${REMOTE_HOST} << EOF
docker run -d \\
  --name ${CONTAINER_NAME} \\
  --restart unless-stopped \\
  -p ${BACKEND_PORT}:8000 \\
  -v /root/nexara_models:/app/models:ro \\
  -e TF_CPP_MIN_LOG_LEVEL=2 \\
  ${IMAGE_NAME}
EOF

echo ""
echo "Step 9: Waiting for container to be healthy..."
sleep 10

echo ""
echo "Step 10: Checking container status..."
sshpass -p "${REMOTE_PASSWORD}" ssh ${REMOTE_USER}@${REMOTE_HOST} << EOF
docker ps | grep ${CONTAINER_NAME}
echo ""
echo "Checking logs..."
docker logs ${CONTAINER_NAME} --tail 50
EOF

echo ""
echo "Step 11: Testing API endpoints..."
echo "Testing health endpoint..."
sshpass -p "${REMOTE_PASSWORD}" ssh ${REMOTE_USER}@${REMOTE_HOST} << EOF
curl -s http://localhost:${BACKEND_PORT}/api/health | python3 -m json.tool || echo "Health check failed"
echo ""
echo "Testing models endpoint..."
curl -s http://localhost:${BACKEND_PORT}/api/models | python3 -m json.tool || echo "Models check failed"
EOF

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "Backend API: http://${REMOTE_HOST}:${BACKEND_PORT}"
echo "Health Check: http://${REMOTE_HOST}:${BACKEND_PORT}/api/health"
echo "API Docs: http://${REMOTE_HOST}:${BACKEND_PORT}/api/docs"
echo "Models Info: http://${REMOTE_HOST}:${BACKEND_PORT}/api/models"
echo ""
echo "Container: ${CONTAINER_NAME}"
echo "Image: ${IMAGE_NAME}"
echo "Models Path: /root/nexara_models (mounted to /app/models)"
echo ""
echo "To view logs:"
echo "  ssh root@${REMOTE_HOST} 'docker logs -f ${CONTAINER_NAME}'"
echo ""

# Cleanup
rm -f /tmp/nexara_vision_prod.tar.gz

echo "Done!"
