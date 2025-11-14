#!/bin/bash

# ============================================================================
# Violence Detection Web App - Deployment Script
# ============================================================================

set -e

echo "=========================================="
echo "Violence Detection - Deployment Script"
echo "=========================================="
echo ""

# Configuration
APP_NAME="violence-detection"
APP_PORT=8000
DOMAIN="violence.nexaratech.io"
SERVER_IP="31.57.166.18"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}→${NC} $1"
}

# Check if running on production server
if [ "$1" == "production" ]; then
    print_info "Deploying to PRODUCTION server..."

    # Stop existing container
    print_info "Stopping existing container..."
    docker stop $APP_NAME 2>/dev/null || true
    docker rm $APP_NAME 2>/dev/null || true

    # Build image
    print_info "Building Docker image..."
    docker build -t $APP_NAME .

    # Run container
    print_info "Starting container..."
    docker run -d \
        --name $APP_NAME \
        -p $APP_PORT:8000 \
        -v /home/admin/Downloads/best_model.h5:/home/admin/Downloads/best_model.h5:ro \
        --restart unless-stopped \
        $APP_NAME

    print_success "Container started on port $APP_PORT"

    # Setup nginx reverse proxy (if not exists)
    NGINX_CONFIG="/etc/nginx/sites-available/$APP_NAME"

    if [ ! -f "$NGINX_CONFIG" ]; then
        print_info "Setting up nginx reverse proxy..."

        sudo tee $NGINX_CONFIG > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN;

    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:$APP_PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Timeouts for video processing
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;
        send_timeout 600;
    }
}
EOF

        sudo ln -sf $NGINX_CONFIG /etc/nginx/sites-enabled/
        sudo nginx -t && sudo systemctl reload nginx

        print_success "Nginx configured"
    else
        print_info "Nginx already configured"
    fi

    # Show status
    echo ""
    echo "=========================================="
    echo "Deployment Complete!"
    echo "=========================================="
    echo ""
    echo "Application URL: http://$DOMAIN"
    echo "Local URL: http://localhost:$APP_PORT"
    echo ""
    echo "Container logs: docker logs -f $APP_NAME"
    echo "Container status: docker ps | grep $APP_NAME"
    echo ""

elif [ "$1" == "local" ]; then
    print_info "Starting LOCAL development server..."

    # Check if model exists
    if [ ! -f "/home/admin/Downloads/best_model.h5" ]; then
        print_error "Model not found at /home/admin/Downloads/best_model.h5"
        exit 1
    fi

    # Install dependencies
    print_info "Installing dependencies..."
    pip install -r requirements.txt

    # Run locally
    print_info "Starting FastAPI server..."
    cd backend && python app.py

else
    echo "Usage: $0 [local|production]"
    echo ""
    echo "  local       - Run locally for testing"
    echo "  production  - Deploy to production server with Docker + nginx"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh local        # Test locally"
    echo "  ./deploy.sh production   # Deploy to production"
    exit 1
fi
