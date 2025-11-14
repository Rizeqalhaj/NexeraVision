# Deployment Quick Start Guide

Quick reference for deploying the Nexara Vision project.

## 🚀 Quick Deploy

### Deploy to Staging (Development Branch)
```bash
git checkout development
git add .
git commit -m "Your changes"
git push origin development
```
**Result**: Automatically deploys to `http://stagingvision.nexara.io`

### Deploy to Production (Main Branch)
```bash
git checkout main
git merge development
git push origin main
```
**Result**: Automatically deploys to `https://vision.nexaratech.io`

## 📋 GitHub Secrets Required

Set these in **GitHub** → **Settings** → **Secrets and variables** → **Actions**:

| Secret | Example Value |
|--------|---------------|
| `SSH_PRIVATE_KEY` | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `SERVER_IP` | `31.57.166.18` |
| `SSH_USER` | `admin` |
| `STAGING_DOMAIN` | `stagingvision.nexara.io` |
| `PRODUCTION_DOMAIN` | `vision.nexaratech.io` |

## 🔧 Manual Deployment

### Staging
```bash
SSH_KEY_PATH=~/.ssh/nexara-deploy-key ./scripts/deploy-staging.sh
```

### Production
```bash
SSH_KEY_PATH=~/.ssh/nexara-deploy-key ./scripts/deploy-production.sh
```

## 🌳 Branch Strategy

```
development → stagingvision.nexara.io (port 8001)
    ↓
   main → vision.nexaratech.io (port 8000)
```

## 🔍 Monitoring

### View Deployment Status
- GitHub Actions: `https://github.com/YOUR_USERNAME/NexeraVision/actions`

### Server Logs
```bash
ssh admin@31.57.166.18

# Staging logs
docker logs -f violence-detection-staging

# Production logs
docker logs -f violence-detection
```

### Container Status
```bash
# List running containers
docker ps

# Check specific container
docker stats violence-detection
```

## ⚡ Common Commands

```bash
# Check current branch
git branch

# View recent commits
git log --oneline -5

# Pull latest changes
git pull origin development

# Merge development to main
git checkout main
git merge development

# Rollback (on server)
docker images | grep backup
docker run -d --name violence-detection -p 8000:8000 violence-detection:backup-TIMESTAMP
```

## 🆘 Troubleshooting

### Deployment Failed
1. Check GitHub Actions logs
2. Verify GitHub secrets
3. Test SSH connection: `ssh admin@31.57.166.18`

### Container Not Running
```bash
docker ps -a
docker logs violence-detection
docker restart violence-detection
```

### Port Conflicts
```bash
sudo lsof -i :8000
sudo lsof -i :8001
```

## 📚 Full Documentation

See [CI-CD-SETUP.md](./CI-CD-SETUP.md) for complete setup instructions.

## 🎯 Production Checklist

Before deploying to production:

- [ ] Test thoroughly on staging
- [ ] Database migrations completed
- [ ] Environment variables configured
- [ ] SSL certificate installed
- [ ] Backup created
- [ ] Team notified
- [ ] Monitoring enabled

## 🔐 Security Notes

- Never commit SSH keys
- Rotate secrets regularly
- Use branch protection
- Require PR reviews
- Monitor access logs

---

**Environments:**
- **Staging**: http://stagingvision.nexara.io (Port 8001)
- **Production**: https://vision.nexaratech.io (Port 8000)

**Server**: 31.57.166.18
