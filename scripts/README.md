# Deployment Scripts

This directory contains helper scripts for deploying the Nexara Vision application.

## Scripts Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy-staging.sh` | Manual staging deployment | `SSH_KEY_PATH=~/.ssh/key ./deploy-staging.sh` |
| `deploy-production.sh` | Manual production deployment | `SSH_KEY_PATH=~/.ssh/key ./deploy-production.sh` |

## Prerequisites

1. SSH access to the server (31.57.166.18)
2. SSH private key for authentication
3. Docker installed on the server
4. Git installed on the server

## Usage

### Deploy to Staging

```bash
# Set the SSH key path
export SSH_KEY_PATH=~/.ssh/nexara-deploy-key

# Run staging deployment
./scripts/deploy-staging.sh
```

### Deploy to Production

```bash
# Set the SSH key path
export SSH_KEY_PATH=~/.ssh/nexara-deploy-key

# Run production deployment (requires confirmation)
./scripts/deploy-production.sh
```

## Environment Variables

The scripts use the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SSH_KEY_PATH` | (required) | Path to SSH private key |
| `SERVER_IP` | `31.57.166.18` | Server IP address |
| `SSH_USER` | `admin` | SSH username |

## What the Scripts Do

### Staging Deployment (`deploy-staging.sh`)

1. Connects to server via SSH
2. Clones/updates code from `development` branch
3. Stops existing staging container
4. Builds new Docker image
5. Starts staging container on port 8001
6. Verifies deployment

### Production Deployment (`deploy-production.sh`)

1. Asks for confirmation
2. Connects to server via SSH
3. Clones/updates code from `main` branch
4. Creates backup of current deployment
5. Stops existing production container
6. Builds new Docker image
7. Starts production container on port 8000
8. Performs health check
9. Rolls back if health check fails

## Troubleshooting

### SSH Connection Failed

**Problem**: Can't connect to server

**Solution**:
```bash
# Test SSH connection
ssh -i ~/.ssh/nexara-deploy-key admin@31.57.166.18

# Check SSH key permissions
chmod 600 ~/.ssh/nexara-deploy-key
```

### Permission Denied

**Problem**: SSH key not recognized

**Solution**:
```bash
# Add public key to server
ssh-copy-id -i ~/.ssh/nexara-deploy-key.pub admin@31.57.166.18
```

### Docker Build Failed

**Problem**: Docker image build fails

**Solution**:
```bash
# SSH into server and check logs
ssh admin@31.57.166.18
cd ~/nexara-vision-staging/web_prototype
docker build -t test .
```

## Automated vs Manual Deployment

| Method | Trigger | Use Case |
|--------|---------|----------|
| **GitHub Actions** (Automated) | Git push | Normal workflow |
| **Manual Scripts** (This directory) | Command line | Emergency fixes, testing |

## Security Notes

- **Never commit SSH keys** to the repository
- Keep `SSH_KEY_PATH` secure
- Use strong passwords for server access
- Rotate SSH keys periodically
- Use different keys for staging and production (recommended)

## Quick Reference

```bash
# Make scripts executable
chmod +x scripts/*.sh

# View script contents
cat scripts/deploy-staging.sh

# Test SSH connection
ssh -i ~/.ssh/nexara-deploy-key admin@31.57.166.18 "echo 'Connection successful'"

# Check running containers on server
ssh -i ~/.ssh/nexara-deploy-key admin@31.57.166.18 "docker ps"
```

## Related Documentation

- [CI/CD Setup Guide](../CI-CD-SETUP.md) - Complete CI/CD configuration
- [Deployment Quick Start](../DEPLOYMENT-QUICKSTART.md) - Quick reference guide
- [Web Prototype README](../web_prototype/README.md) - Application documentation

## Support

For issues with deployment:
1. Check GitHub Actions logs
2. Check server logs: `ssh admin@31.57.166.18 "docker logs violence-detection"`
3. Verify network connectivity
4. Ensure Docker is running on server
