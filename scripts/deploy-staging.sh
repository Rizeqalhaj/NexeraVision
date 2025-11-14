#!/bin/bash

# ============================================================================
# Manual Staging Deployment Script
# ============================================================================
# This script can be run manually to deploy to staging
# Usage: ./scripts/deploy-staging.sh

set -e

echo "=========================================="
echo "Manual Staging Deployment"
echo "=========================================="

# Configuration
SERVER_IP="${SERVER_IP:-31.57.166.18}"
SSH_USER="${SSH_USER:-admin}"
STAGING_PORT=8001
STAGING_DOMAIN="stagingvision.nexara.io"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${YELLOW}→${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "web_prototype/Dockerfile" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if SSH key is available
if [ -z "$SSH_KEY_PATH" ]; then
    print_error "SSH_KEY_PATH environment variable not set"
    echo "Usage: SSH_KEY_PATH=/path/to/key ./scripts/deploy-staging.sh"
    exit 1
fi

print_info "Deploying to staging server: $SERVER_IP"
print_info "Staging URL will be: http://$STAGING_DOMAIN"

# Deploy via SSH
ssh -i "$SSH_KEY_PATH" ${SSH_USER}@${SERVER_IP} << 'ENDSSH'
    set -e

    echo "Creating staging directory..."
    mkdir -p ~/nexara-vision-staging
    cd ~/nexara-vision-staging

    echo "Updating code from development branch..."
    if [ -d ".git" ]; then
        git fetch origin
        git reset --hard origin/development
    else
        git clone -b development https://github.com/YOUR_GITHUB_USERNAME/NexeraVision.git .
    fi

    cd web_prototype

    echo "Stopping existing staging container..."
    docker stop violence-detection-staging 2>/dev/null || true
    docker rm violence-detection-staging 2>/dev/null || true

    echo "Building Docker image..."
    docker build -t violence-detection:staging .

    echo "Starting staging container..."
    docker run -d \
        --name violence-detection-staging \
        -p 8001:8000 \
        -v /home/admin/Downloads/best_model.h5:/home/admin/Downloads/best_model.h5:ro \
        --restart unless-stopped \
        violence-detection:staging

    echo "Waiting for container to start..."
    sleep 5

    if docker ps | grep -q violence-detection-staging; then
        echo "=========================================="
        echo "✓ Staging deployment successful!"
        echo "=========================================="
    else
        echo "✗ Deployment failed"
        docker logs violence-detection-staging
        exit 1
    fi
ENDSSH

print_success "Staging deployment complete!"
print_info "Access staging at: http://$STAGING_DOMAIN"
