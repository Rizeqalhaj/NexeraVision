# NexaraVision Deployment Guide

## Port Configuration

### Production (https://vision.nexaratech.io)
- **Branch**: `main`
- **Frontend (Next.js)**: Port 3005
- **Backend (NestJS)**: Port 3006
- **ML Service (Python)**: Port 3007

### Staging (http://stagingvision.nexaratech.io)
- **Branch**: `development`
- **Frontend (Next.js)**: Port 8001
- **Backend (NestJS)**: Port 8002
- **ML Service (Python)**: Port 8003

## Deployment Methods

### Method 1: Automatic Deployment (GitHub Actions)

#### For Staging
```bash
git checkout development
# Make your changes
git add .
git commit -m "feat: your changes"
git push origin development
```

The GitHub Actions workflow will automatically:
1. Deploy to `http://stagingvision.nexaratech.io`
2. Use ports 8001, 8002, 8003
3. Run health checks
4. Report deployment status

#### For Production
```bash
git checkout main
git merge development  # or your feature branch
git push origin main
```

The GitHub Actions workflow will automatically:
1. Deploy to `https://vision.nexaratech.io`
2. Use ports 3005, 3006, 3007
3. Run comprehensive health checks
4. Report deployment status

### Method 2: Manual Deployment

If you need to deploy manually to the server:

```bash
# Copy the deployment script to the server
scp deploy-production-manual.sh admin@31.57.166.18:~

# SSH into the server
ssh admin@31.57.166.18

# Run the deployment script
bash ~/deploy-production-manual.sh
```

Or run it directly:
```bash
ssh admin@31.57.166.18 'bash -s' < deploy-production-manual.sh
```

## Verification

### Check Production
```bash
# Check PM2 processes
ssh admin@31.57.166.18 'pm2 list | grep production'

# Check Docker containers
ssh admin@31.57.166.18 'docker ps | grep production'

# Test endpoints
curl https://vision.nexaratech.io
curl https://vision.nexaratech.io/api
```

### Check Staging
```bash
# Check PM2 processes
ssh admin@31.57.166.18 'pm2 list | grep staging'

# Check Docker containers
ssh admin@31.57.166.18 'docker ps | grep staging'

# Test endpoints
curl http://stagingvision.nexaratech.io
curl http://stagingvision.nexaratech.io/api
```

## Nginx Configuration

The deployment scripts automatically configure nginx, but for reference:

### Production
- Domain: `vision.nexaratech.io`
- SSL: Enabled (Let's Encrypt)
- Frontend proxy: `localhost:3005`
- Backend API proxy: `localhost:3006`
- ML Service: Internal only (`localhost:3007`)

### Staging
- Domain: `stagingvision.nexaratech.io`
- SSL: HTTP only
- Frontend proxy: `localhost:8001`
- Backend API proxy: `localhost:8002`
- ML Service: Internal only (`localhost:8003`)

## Troubleshooting

### Services not starting
```bash
# Check PM2 logs
ssh admin@31.57.166.18 'pm2 logs nexara-vision-frontend-production'
ssh admin@31.57.166.18 'pm2 logs nexara-vision-backend-production'

# Check Docker logs
ssh admin@31.57.166.18 'docker logs nexara-ml-service-production'
```

### Port conflicts
```bash
# Check what's using the ports
ssh admin@31.57.166.18 'sudo netstat -tlnp | grep -E ":(3005|3006|3007|8001|8002|8003)"'

# Kill processes using the ports if needed
ssh admin@31.57.166.18 'sudo fuser -k 3005/tcp'
```

### Nginx issues
```bash
# Test nginx configuration
ssh admin@31.57.166.18 'sudo nginx -t'

# Reload nginx
ssh admin@31.57.166.18 'sudo systemctl reload nginx'

# Check nginx logs
ssh admin@31.57.166.18 'sudo tail -f /var/log/nginx/error.log'
```

## Database Configuration

### Production Database
- Name: `nexara_vision_production`
- Port: 5432
- Connection: `postgresql://postgres:E$$athecode006@localhost:5432/nexara_vision_production`

### Staging Database
- Name: `nexara_vision_staging`
- Port: 5432
- Connection: `postgresql://postgres:E$$athecode006@localhost:5432/nexara_vision_staging`

## Quick Commands

### Restart Production Services
```bash
ssh admin@31.57.166.18 'pm2 restart nexara-vision-frontend-production nexara-vision-backend-production'
ssh admin@31.57.166.18 'docker restart nexara-ml-service-production'
```

### Restart Staging Services
```bash
ssh admin@31.57.166.18 'pm2 restart nexara-vision-frontend-staging nexara-vision-backend-staging'
ssh admin@31.57.166.18 'docker restart nexara-ml-service-staging'
```

### View All Services
```bash
ssh admin@31.57.166.18 'pm2 list && docker ps'
```

## Summary

- ✅ Production workflow updated to use ports 3005, 3006, 3007
- ✅ Staging workflow using ports 8001, 8002, 8003
- ✅ Nginx configurations included in deployment scripts
- ✅ Health checks implemented for all services
- ✅ Automatic rollback on failure

When you push to the `main` branch, the GitHub Actions workflow will automatically deploy to production with the correct port configuration.
