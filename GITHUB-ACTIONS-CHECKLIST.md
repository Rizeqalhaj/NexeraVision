# GitHub Actions CI/CD Setup Checklist

Use this checklist to ensure your CI/CD pipeline is properly configured.

## ✅ Prerequisites

- [ ] GitHub repository created
- [ ] Server access (31.57.166.18) with SSH
- [ ] Docker installed on server
- [ ] Nginx installed on server
- [ ] Git installed on server
- [ ] Domain DNS configured (vision.nexaratech.io, stagingvision.nexara.io)

## ✅ SSH Key Setup

- [ ] SSH key pair generated (`ssh-keygen -t ed25519 -C "github-actions-deploy"`)
- [ ] Public key added to server (`~/.ssh/authorized_keys`)
- [ ] Private key saved securely (DO NOT commit to repo)
- [ ] SSH connection tested (`ssh -i ~/.ssh/nexara-deploy-key admin@31.57.166.18`)

## ✅ GitHub Repository Configuration

### Branches
- [ ] `main` or `master` branch exists
- [ ] `development` branch created
- [ ] Branch protection rules configured (optional but recommended)

### GitHub Secrets
Navigate to: **Settings** → **Secrets and variables** → **Actions**

- [ ] `SSH_PRIVATE_KEY` - Private SSH key content
- [ ] `SERVER_IP` - `31.57.166.18`
- [ ] `SSH_USER` - Your SSH username (e.g., `admin`)
- [ ] `STAGING_DOMAIN` - `stagingvision.nexara.io`
- [ ] `PRODUCTION_DOMAIN` - `vision.nexaratech.io`

### GitHub Environment (Optional)
Navigate to: **Settings** → **Environments**

- [ ] `production` environment created
- [ ] Required reviewers added
- [ ] Deployment protection rules configured

## ✅ Server Setup

### Directory Structure
SSH into server and verify:

```bash
ssh admin@31.57.166.18

# Check Docker
docker --version

# Check nginx
nginx -v

# Check Git
git --version
```

- [ ] Docker is installed and running
- [ ] Nginx is installed and running
- [ ] Git is installed
- [ ] Model file exists at `/home/admin/Downloads/best_model.h5`

### Nginx Configuration

- [ ] Nginx is running: `sudo systemctl status nginx`
- [ ] Port 80 is open and accessible
- [ ] DNS records point to server IP

## ✅ Workflow Files

- [ ] `.github/workflows/staging.yml` exists
- [ ] `.github/workflows/production.yml` exists
- [ ] Workflow files are in the correct format (YAML)
- [ ] Workflow files reference correct branches

## ✅ Testing

### Test SSH Connection
```bash
ssh -i ~/.ssh/nexara-deploy-key admin@31.57.166.18 "echo 'Connection successful'"
```
- [ ] SSH connection successful

### Test Staging Deployment
```bash
# Make a test change
git checkout development
echo "# Test" >> README.md
git add README.md
git commit -m "Test staging deployment"
git push origin development
```
- [ ] GitHub Actions workflow triggered
- [ ] Workflow completed successfully
- [ ] Staging site is accessible at `http://stagingvision.nexara.io`

### Test Production Deployment
```bash
# Merge to main
git checkout main
git merge development
git push origin main
```
- [ ] GitHub Actions workflow triggered
- [ ] Workflow completed successfully (or waiting for approval if environment protection is enabled)
- [ ] Production site is accessible at `https://vision.nexaratech.io`

## ✅ Verification

### GitHub Actions
- [ ] Navigate to **Actions** tab in GitHub
- [ ] Recent workflows show successful runs (green checkmarks)
- [ ] No errors in workflow logs

### Server Verification
```bash
ssh admin@31.57.166.18

# Check staging container
docker ps | grep violence-detection-staging

# Check production container
docker ps | grep violence-detection

# Check nginx
sudo systemctl status nginx
```

- [ ] Staging container is running (port 8001)
- [ ] Production container is running (port 8000)
- [ ] Nginx is serving both domains

### Website Verification

- [ ] Staging site loads: `http://stagingvision.nexara.io`
- [ ] Production site loads: `https://vision.nexaratech.io`
- [ ] Video upload works on staging
- [ ] Video upload works on production
- [ ] Model predictions are working

## ✅ Security

- [ ] SSH private key is NOT committed to repository
- [ ] `.gitignore` includes `*.key`, `*.pem`, `.env*`
- [ ] GitHub secrets are properly configured
- [ ] Server firewall allows ports 80, 443, 22
- [ ] SSH password authentication is disabled (key-only)
- [ ] Strong server passwords are used

## ✅ Documentation

- [ ] [CI-CD-SETUP.md](./CI-CD-SETUP.md) reviewed
- [ ] [DEPLOYMENT-QUICKSTART.md](./DEPLOYMENT-QUICKSTART.md) reviewed
- [ ] Team members trained on deployment process
- [ ] Emergency rollback procedure documented

## ✅ Monitoring & Maintenance

- [ ] GitHub Actions notifications enabled
- [ ] Server monitoring configured
- [ ] Log rotation configured
- [ ] Backup strategy in place
- [ ] SSL certificates installed (for HTTPS)
- [ ] Auto-renewal configured for SSL

## 🔧 Quick Commands Reference

### View GitHub Actions Logs
```bash
# Using GitHub CLI
gh run list
gh run view --log
```

### View Server Logs
```bash
# Staging
docker logs -f violence-detection-staging

# Production
docker logs -f violence-detection
```

### Manual Deployment
```bash
# Staging
SSH_KEY_PATH=~/.ssh/nexara-deploy-key ./scripts/deploy-staging.sh

# Production
SSH_KEY_PATH=~/.ssh/nexara-deploy-key ./scripts/deploy-production.sh
```

### Rollback Production
```bash
ssh admin@31.57.166.18
docker images | grep violence-detection:backup
docker stop violence-detection
docker rm violence-detection
docker run -d --name violence-detection -p 8000:8000 \
  -v /home/admin/Downloads/best_model.h5:/home/admin/Downloads/best_model.h5:ro \
  --restart unless-stopped \
  violence-detection:backup-TIMESTAMP
```

## 📋 Common Issues & Solutions

### Issue: GitHub Actions fails with "Permission denied (publickey)"
**Solution**: Verify SSH_PRIVATE_KEY secret is correctly configured
```bash
cat ~/.ssh/nexara-deploy-key  # Copy this exact content to GitHub secret
```

### Issue: Container not starting on server
**Solution**: Check container logs and ensure model file exists
```bash
docker logs violence-detection
ls -lh /home/admin/Downloads/best_model.h5
```

### Issue: Nginx 502 Bad Gateway
**Solution**: Ensure container is running on correct port
```bash
docker ps
curl http://localhost:8000
```

### Issue: Domain not resolving
**Solution**: Check DNS configuration
```bash
nslookup vision.nexaratech.io
ping stagingvision.nexara.io
```

## 🎯 Next Steps After Setup

1. **Configure PostgreSQL Database**
   - Install PostgreSQL on server
   - Create databases for staging and production
   - Update environment variables
   - Add database migrations to CI/CD

2. **Set Up SSL Certificates**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d vision.nexaratech.io
   sudo certbot --nginx -d stagingvision.nexara.io
   ```

3. **Configure Monitoring**
   - Set up application monitoring
   - Configure error tracking (e.g., Sentry)
   - Set up uptime monitoring

4. **Add Automated Tests**
   - Unit tests in CI pipeline
   - Integration tests
   - End-to-end tests

## ✨ Success Criteria

Your CI/CD pipeline is fully operational when:

- ✅ Pushing to `development` automatically deploys to staging
- ✅ Pushing to `main` automatically deploys to production
- ✅ Both environments are accessible via their respective domains
- ✅ Application functions correctly in both environments
- ✅ Deployment failures trigger automatic rollback
- ✅ Team can monitor deployments via GitHub Actions
- ✅ Manual deployment scripts work as backup

---

**Last Updated**: 2025-11-14
**Maintained By**: Nexara Tech Team
**Support**: Check GitHub Actions logs or server logs for issues
