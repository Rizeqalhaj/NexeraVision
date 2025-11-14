#!/bin/bash

# ============================================================================
# Manual Production Deployment Script
# ============================================================================
# This script can be run manually to deploy to production
# Usage: ./scripts/deploy-production.sh

set -e

echo "=========================================="
echo "Manual Production Deployment"
echo "=========================================="

# Configuration
SERVER_IP="${SERVER_IP:-31.57.166.18}"
SSH_USER="${SSH_USER:-admin}"
PRODUCTION_PORT=8000
PRODUCTION_DOMAIN="vision.nexaratech.io"

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
    echo "Usage: SSH_KEY_PATH=/path/to/key ./scripts/deploy-production.sh"
    exit 1
fi

# Confirmation prompt
print_error "WARNING: This will deploy to PRODUCTION!"
read -p "Are you sure you want to continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    print_info "Deployment cancelled"
    exit 0
fi

print_info "Deploying to production server: $SERVER_IP"
print_info "Production URL: https://$PRODUCTION_DOMAIN"

# Deploy via SSH
ssh -i "$SSH_KEY_PATH" ${SSH_USER}@${SERVER_IP} << 'ENDSSH'
    set -e

    echo "Creating production directory..."
    mkdir -p ~/nexara-vision-production
    cd ~/nexara-vision-production

    echo "Updating code from main branch..."
    if [ -d ".git" ]; then
        git fetch origin
        git reset --hard origin/main
    else
        git clone -b main https://github.com/YOUR_GITHUB_USERNAME/NexeraVision.git .
    fi

    cd web_prototype

    echo "Creating backup of current deployment..."
    docker commit violence-detection violence-detection:backup-$(date +%Y%m%d-%H%M%S) 2>/dev/null || true

    echo "Stopping existing production container..."
    docker stop violence-detection 2>/dev/null || true
    docker rm violence-detection 2>/dev/null || true

    echo "Building Docker image..."
    docker build -t violence-detection:latest -t violence-detection:production .

    echo "Starting production container..."
    docker run -d \
        --name violence-detection \
        -p 8000:8000 \
        -v /home/admin/Downloads/best_model.h5:/home/admin/Downloads/best_model.h5:ro \
        --restart unless-stopped \
        violence-detection:production

    echo "Waiting for container to start..."
    sleep 5

    echo "Performing health check..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/ > /dev/null 2>&1; then
            echo "✓ Health check passed"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "✗ Health check failed"
            docker logs violence-detection
            exit 1
        fi
        sleep 2
    done

    if docker ps | grep -q violence-detection; then
        echo "=========================================="
        echo "✓ Production deployment successful!"
        echo "=========================================="

        # Clean up old backups
        docker images | grep violence-detection:backup | tail -n +4 | awk '{print $3}' | xargs -r docker rmi 2>/dev/null || true
    else
        echo "✗ Deployment failed"
        docker logs violence-detection

        # Rollback
        echo "Attempting rollback..."
        BACKUP_IMAGE=$(docker images | grep violence-detection:backup | head -n 1 | awk '{print $1":"$2}')
        if [ ! -z "$BACKUP_IMAGE" ]; then
            echo "Rolling back to $BACKUP_IMAGE"
            docker run -d \
                --name violence-detection \
                -p 8000:8000 \
                -v /home/admin/Downloads/best_model.h5:/home/admin/Downloads/best_model.h5:ro \
                --restart unless-stopped \
                $BACKUP_IMAGE
        fi
        exit 1
    fi
ENDSSH

print_success "Production deployment complete!"
print_info "Access production at: https://$PRODUCTION_DOMAIN"
