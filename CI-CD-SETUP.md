# CI/CD Pipeline Setup Guide

This guide will help you set up the complete CI/CD pipeline for the Nexara Vision project.

## Overview

The CI/CD pipeline automatically deploys your application:
- **Development branch** → **Staging environment** (automatic)
- **Main branch** → **Production environment** (automatic)

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Development    │─────▶│  GitHub Actions  │─────▶│  Staging Server │
│  Branch         │      │  (Auto Deploy)   │      │  Port 8001      │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                     stagingvision.nexara.io

┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Main/Master    │─────▶│  GitHub Actions  │─────▶│  Production     │
│  Branch         │      │  (Auto Deploy)   │      │  Port 8000      │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                     vision.nexaratech.io
```

## Prerequisites

1. GitHub repository for the project
2. Server with Docker and nginx installed
3. SSH access to the server
4. Domain/subdomain configured for staging and production

## Step 1: Server Setup

### 1.1 Install Required Software

SSH into your server (`31.57.166.18`) and install:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install nginx
sudo apt install nginx -y

# Install git
sudo apt install git -y
```

### 1.2 Create SSH Key for GitHub Actions

On your **local machine**, generate an SSH key pair:

```bash
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/nexara-deploy-key
```

This creates:
- **Private key**: `~/.ssh/nexara-deploy-key` (keep this secret!)
- **Public key**: `~/.ssh/nexara-deploy-key.pub`

### 1.3 Add Public Key to Server

Copy the public key to your server:

```bash
ssh-copy-id -i ~/.ssh/nexara-deploy-key.pub admin@31.57.166.18
```

Or manually add it to `~/.ssh/authorized_keys` on the server.

### 1.4 Configure DNS

Set up DNS records for your domains:

| Type | Name    | Value          |
|------|---------|----------------|
| A    | vision  | 31.57.166.18   |
| A    | staging.vision | 31.57.166.18 |

## Step 2: GitHub Repository Setup

### 2.1 Create Required Branches

```bash
# Create development branch
git checkout -b development
git push -u origin development

# Main branch should already exist
git checkout main
```

### 2.2 Configure GitHub Secrets

Go to your GitHub repository:
1. Navigate to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add the following secrets:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `SSH_PRIVATE_KEY` | Contents of `~/.ssh/nexara-deploy-key` | Private SSH key for deployment |
| `SERVER_IP` | `31.57.166.18` | Server IP address |
| `SSH_USER` | `admin` (or your username) | SSH username |
| `STAGING_DOMAIN` | `stagingvision.nexara.io` | Staging domain |
| `PRODUCTION_DOMAIN` | `vision.nexaratech.io` | Production domain |

**To get the SSH private key contents:**

```bash
cat ~/.ssh/nexara-deploy-key
```

Copy the entire output (including `-----BEGIN OPENSSH PRIVATE KEY-----` and `-----END OPENSSH PRIVATE KEY-----`).

### 2.3 Create GitHub Environment (Optional but Recommended)

For production deployments with approval:

1. Go to **Settings** → **Environments**
2. Click **New environment**
3. Name it `production`
4. Add protection rules:
   - ✅ Required reviewers (add yourself)
   - ✅ Wait timer (optional): 5 minutes
5. Click **Save protection rules**

## Step 3: Branching Strategy

### Development Workflow

```bash
# 1. Create a feature branch from development
git checkout development
git pull origin development
git checkout -b feature/my-new-feature

# 2. Make your changes and commit
git add .
git commit -m "Add new feature"

# 3. Push to GitHub
git push origin feature/my-new-feature

# 4. Create a Pull Request to development branch
# Review and merge PR

# 5. When merged to development, it auto-deploys to staging!
```

### Production Deployment Workflow

```bash
# 1. After testing on staging, merge development to main
git checkout main
git pull origin main
git merge development

# 2. Push to production
git push origin main

# 3. Automatically deploys to production!
```

## Step 4: Workflow Files

The following workflow files are already created:

### `.github/workflows/staging.yml`
- Triggers on push to `development` branch
- Deploys to staging server (port 8001)
- Available at: `http://stagingvision.nexara.io`

### `.github/workflows/production.yml`
- Triggers on push to `main` or `master` branch
- Deploys to production server (port 8000)
- Available at: `https://vision.nexaratech.io`
- Includes health checks and automatic rollback

## Step 5: Manual Deployment Scripts

For manual deployments or testing:

### Deploy to Staging
```bash
SSH_KEY_PATH=~/.ssh/nexara-deploy-key ./scripts/deploy-staging.sh
```

### Deploy to Production
```bash
SSH_KEY_PATH=~/.ssh/nexara-deploy-key ./scripts/deploy-production.sh
```

## Step 6: Testing the Pipeline

### Test Staging Deployment

```bash
# Make a change
echo "# Test change" >> README.md
git add README.md
git commit -m "Test staging deployment"
git push origin development

# Watch the deployment in GitHub Actions
# Visit: https://github.com/YOUR_USERNAME/NexeraVision/actions
```

### Test Production Deployment

```bash
# Merge to main
git checkout main
git merge development
git push origin main

# Watch the deployment in GitHub Actions
```

## Step 7: Monitoring and Logs

### View Deployment Logs in GitHub

1. Go to **Actions** tab in your repository
2. Click on the latest workflow run
3. View logs for each step

### View Container Logs on Server

```bash
# SSH into server
ssh admin@31.57.166.18

# View staging logs
docker logs -f violence-detection-staging

# View production logs
docker logs -f violence-detection

# View nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Check Running Containers

```bash
# List all running containers
docker ps

# Check container status
docker stats violence-detection
docker stats violence-detection-staging
```

## Troubleshooting

### Deployment Fails with SSH Error

**Solution**: Ensure the SSH private key is correctly added to GitHub secrets:
```bash
cat ~/.ssh/nexara-deploy-key | pbcopy  # macOS
cat ~/.ssh/nexara-deploy-key | xclip   # Linux
```

### Container Not Starting

**Solution**: Check container logs:
```bash
docker logs violence-detection
docker logs violence-detection-staging
```

### Nginx Configuration Error

**Solution**: Test nginx configuration:
```bash
sudo nginx -t
sudo systemctl status nginx
```

### Port Already in Use

**Solution**: Check what's using the port:
```bash
sudo lsof -i :8000
sudo lsof -i :8001
```

## Environment Variables

### Staging Environment
- **Port**: 8001
- **Container Name**: `violence-detection-staging`
- **Domain**: `stagingvision.nexara.io`
- **Docker Image**: `violence-detection:staging`

### Production Environment
- **Port**: 8000
- **Container Name**: `violence-detection`
- **Domain**: `vision.nexaratech.io`
- **Docker Image**: `violence-detection:production`

## Security Best Practices

1. ✅ Never commit SSH keys to the repository
2. ✅ Use GitHub Secrets for sensitive data
3. ✅ Enable branch protection rules
4. ✅ Require PR reviews before merging
5. ✅ Use environment-specific configurations
6. ✅ Regularly rotate SSH keys
7. ✅ Monitor deployment logs

## Quick Reference Commands

```bash
# Check deployment status
git status
git log --oneline

# View branches
git branch -a

# Switch branches
git checkout development
git checkout main

# View GitHub Actions locally (using act)
gh run list
gh run view

# SSH into server
ssh admin@31.57.166.18

# Restart containers
docker restart violence-detection
docker restart violence-detection-staging

# View all logs
docker logs violence-detection --tail 100
```

## Next Steps

1. ✅ Set up PostgreSQL database (as per your requirements)
2. ✅ Configure environment-specific database connections
3. ✅ Add database migration scripts to CI/CD
4. ✅ Set up SSL certificates (Let's Encrypt)
5. ✅ Configure monitoring and alerting
6. ✅ Add automated testing to pipeline

## Support

If you encounter issues:
1. Check GitHub Actions logs
2. Check server logs: `docker logs violence-detection`
3. Verify GitHub secrets are correctly configured
4. Ensure SSH access is working: `ssh admin@31.57.166.18`

---

**Important Notes:**
- The model file at `/home/admin/Downloads/best_model.h5` must exist on the server
- Ensure Docker has sufficient disk space for images
- Both staging and production use the same model file
- Nginx must be running and configured correctly
