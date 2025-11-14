#!/bin/bash
# Test Nexara Vision Enterprise locally before deployment

set -e

echo "========================================="
echo "Nexara Vision Local Testing"
echo "========================================="
echo ""

# Configuration
TEST_PORT=8005
MODELS_PATH="/home/admin/Desktop/NexaraVision/downloaded_models"
CONTAINER_NAME="nexara-vision-test"
IMAGE_NAME="nexara-vision:test"

echo "Step 1: Building test Docker image..."
docker build -f Dockerfile.production -t ${IMAGE_NAME} .

echo ""
echo "Step 2: Stopping any existing test container..."
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

echo ""
echo "Step 3: Starting test container..."
docker run -d \
  --name ${CONTAINER_NAME} \
  -p ${TEST_PORT}:8000 \
  -v ${MODELS_PATH}:/app/models:ro \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  ${IMAGE_NAME}

echo ""
echo "Step 4: Waiting for container to start..."
sleep 15

echo ""
echo "Step 5: Checking container status..."
docker ps | grep ${CONTAINER_NAME} || {
    echo "ERROR: Container failed to start"
    docker logs ${CONTAINER_NAME}
    exit 1
}

echo ""
echo "Step 6: Viewing startup logs..."
docker logs ${CONTAINER_NAME}

echo ""
echo "Step 7: Testing API endpoints..."
echo ""

echo "Testing health endpoint..."
curl -s http://localhost:${TEST_PORT}/api/health | python3 -m json.tool || {
    echo "ERROR: Health check failed"
    docker logs ${CONTAINER_NAME}
    exit 1
}

echo ""
echo "Testing models endpoint..."
curl -s http://localhost:${TEST_PORT}/api/models | python3 -m json.tool || {
    echo "ERROR: Models check failed"
    exit 1
}

echo ""
echo "Testing info endpoint..."
curl -s http://localhost:${TEST_PORT}/api/info | python3 -m json.tool || {
    echo "ERROR: Info check failed"
    exit 1
}

echo ""
echo "========================================="
echo "Local Test Complete!"
echo "========================================="
echo ""
echo "Container is running and healthy!"
echo ""
echo "Test URLs:"
echo "  Health: http://localhost:${TEST_PORT}/api/health"
echo "  Models: http://localhost:${TEST_PORT}/api/models"
echo "  Info:   http://localhost:${TEST_PORT}/api/info"
echo "  Docs:   http://localhost:${TEST_PORT}/api/docs"
echo ""
echo "To test video upload:"
echo "  curl -X POST http://localhost:${TEST_PORT}/upload -F 'video=@test.mp4'"
echo ""
echo "To view logs:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo ""
echo "To stop test container:"
echo "  docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}"
echo ""
