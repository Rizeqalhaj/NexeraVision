#!/bin/bash

# ============================================================================
# SSH Key Setup Helper Script
# ============================================================================
# This script helps generate SSH keys for GitHub Actions deployment

set -e

echo "=========================================="
echo "SSH Key Setup for GitHub Actions"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

print_step() {
    echo -e "${BLUE}➜${NC} $1"
}

# Configuration
SSH_KEY_NAME="nexara-deploy-key"
SSH_KEY_PATH="$HOME/.ssh/$SSH_KEY_NAME"
SERVER_IP="31.57.166.18"

# Step 1: Generate SSH Key
print_step "Step 1: Generate SSH Key"
echo ""

if [ -f "$SSH_KEY_PATH" ]; then
    print_info "SSH key already exists at: $SSH_KEY_PATH"
    read -p "Do you want to generate a new key? (yes/no): " regenerate
    if [ "$regenerate" != "yes" ]; then
        print_info "Using existing key"
    else
        print_info "Generating new SSH key..."
        ssh-keygen -t ed25519 -C "github-actions-deploy" -f "$SSH_KEY_PATH" -N ""
        print_success "New SSH key generated"
    fi
else
    print_info "Generating SSH key..."
    ssh-keygen -t ed25519 -C "github-actions-deploy" -f "$SSH_KEY_PATH" -N ""
    print_success "SSH key generated at: $SSH_KEY_PATH"
fi

echo ""

# Step 2: Display Public Key
print_step "Step 2: Add Public Key to Server"
echo ""
print_info "Public key content:"
echo ""
echo "────────────────────────────────────────"
cat "${SSH_KEY_PATH}.pub"
echo "────────────────────────────────────────"
echo ""

read -p "Enter your server SSH username (default: admin): " SSH_USER
SSH_USER=${SSH_USER:-admin}

print_info "Attempting to copy public key to server..."
print_info "You will be prompted for your server password"
echo ""

if ssh-copy-id -i "${SSH_KEY_PATH}.pub" ${SSH_USER}@${SERVER_IP}; then
    print_success "Public key added to server"
else
    print_error "Failed to automatically add key"
    echo ""
    print_info "Please manually add the public key above to:"
    echo "  ${SSH_USER}@${SERVER_IP}:~/.ssh/authorized_keys"
fi

echo ""

# Step 3: Test Connection
print_step "Step 3: Test SSH Connection"
echo ""

print_info "Testing SSH connection..."
if ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no ${SSH_USER}@${SERVER_IP} "echo 'Connection successful'"; then
    print_success "SSH connection successful!"
else
    print_error "SSH connection failed"
    echo ""
    print_info "Troubleshooting:"
    echo "  1. Ensure the public key is in ~/.ssh/authorized_keys on the server"
    echo "  2. Check SSH service is running: sudo systemctl status ssh"
    echo "  3. Verify firewall allows SSH: sudo ufw status"
    exit 1
fi

echo ""

# Step 4: Display Private Key for GitHub
print_step "Step 4: Add Private Key to GitHub Secrets"
echo ""

print_info "Copy the private key below to GitHub:"
echo ""
echo "1. Go to: GitHub Repository → Settings → Secrets and variables → Actions"
echo "2. Click: New repository secret"
echo "3. Name: SSH_PRIVATE_KEY"
echo "4. Value: Copy the content below (including BEGIN and END lines)"
echo ""
echo "────────────────────────────────────────"
cat "$SSH_KEY_PATH"
echo "────────────────────────────────────────"
echo ""

print_info "IMPORTANT: Keep this private key secure!"
print_info "Never commit it to your repository!"

echo ""

# Step 5: Summary
print_step "Step 5: Next Steps"
echo ""

echo "Add these GitHub Secrets:"
echo ""
echo "  Secret Name          | Value"
echo "  ─────────────────────|──────────────────────────────"
echo "  SSH_PRIVATE_KEY      | (content shown above)"
echo "  SERVER_IP            | ${SERVER_IP}"
echo "  SSH_USER             | ${SSH_USER}"
echo "  STAGING_DOMAIN       | stagingvision.nexara.io"
echo "  PRODUCTION_DOMAIN    | vision.nexaratech.io"
echo ""

print_success "SSH key setup complete!"
echo ""
print_info "Next: Follow the CI-CD-SETUP.md guide to complete the setup"

# Save configuration
CONFIG_FILE="$HOME/.nexara-deploy-config"
cat > "$CONFIG_FILE" <<EOF
# Nexara Vision Deployment Configuration
# Generated: $(date)

SSH_KEY_PATH=$SSH_KEY_PATH
SERVER_IP=$SERVER_IP
SSH_USER=$SSH_USER
STAGING_DOMAIN=stagingvision.nexara.io
PRODUCTION_DOMAIN=vision.nexaratech.io
EOF

print_success "Configuration saved to: $CONFIG_FILE"
echo ""
