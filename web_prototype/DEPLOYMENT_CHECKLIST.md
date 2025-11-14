# ✅ Deployment Checklist for vision.nexaratech.io

## Pre-Deployment Verification

### Local Testing
- [ ] Test model loading: Model exists at `/home/admin/Downloads/best_model.h5` (34MB)
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run local server: `./deploy.sh local`
- [ ] Test upload: Upload sample video at `http://localhost:8000`
- [ ] Verify API: `curl http://localhost:8000/api/health`

### Production Server Preparation (31.57.166.18)

#### 1. DNS Configuration
- [ ] Point `vision.nexaratech.io` A record → `31.57.166.18`
- [ ] Verify DNS propagation: `nslookup vision.nexaratech.io`

#### 2. Server Requirements
```bash
# Check Docker
docker --version  # Should be 20.10+

# Check disk space
df -h  # Need ~10GB free

# Check memory
free -h  # Need 4GB+ RAM

# Check model file
ls -lh /home/admin/Downloads/best_model.h5  # Should show 34MB
```

#### 3. Upload Project
```bash
# From your local machine
scp -r /home/admin/Desktop/NexaraVision/web_prototype admin@31.57.166.18:/home/admin/

# Or using rsync (better for updates)
rsync -avz --progress /home/admin/Desktop/NexaraVision/web_prototype/ admin@31.57.166.18:/home/admin/web_prototype/
```

## Deployment Steps

### Step 1: Deploy Application
```bash
ssh admin@31.57.166.18
cd /home/admin/web_prototype
./deploy.sh production
```

**Expected output**:
```
→ Stopping existing container...
→ Building Docker image...
→ Starting container...
✓ Container started on port 8000
✓ Nginx configured

==========================================
Deployment Complete!
==========================================

Application URL: http://vision.nexaratech.io
Local URL: http://localhost:8000
```

### Step 2: Verify Deployment
```bash
# Check container is running
docker ps | grep violence-detection

# Check logs (should see "Model loaded successfully")
docker logs violence-detection-app | grep "Model loaded"

# Test health endpoint
curl http://localhost:8000/api/health
```

### Step 3: Configure SSL (HTTPS)
```bash
# Install certbot (if not already installed)
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d vision.nexaratech.io

# Select option 2: Redirect HTTP to HTTPS
```

**Expected output**:
```
Congratulations! You have successfully enabled HTTPS on
https://vision.nexaratech.io
```

### Step 4: Final Verification
```bash
# Test HTTPS
curl https://vision.nexaratech.io/api/health

# Test file upload (with sample video)
curl -X POST https://vision.nexaratech.io/api/predict \
  -F "file=@test_video.mp4"
```

### Step 5: Test Web Interface
- [ ] Open browser: `https://vision.nexaratech.io`
- [ ] Upload video via drag & drop
- [ ] Verify result displays correctly
- [ ] Test with different video formats (MP4, AVI, MOV)
- [ ] Test error handling (large file, wrong format)

## Post-Deployment

### Monitoring Setup
```bash
# Create monitoring script
cat > /home/admin/monitor_violence_app.sh << 'EOF'
#!/bin/bash
if ! docker ps | grep -q violence-detection-app; then
    echo "⚠️ Violence detection app is down! Restarting..."
    cd /home/admin/web_prototype
    ./deploy.sh production
    echo "✅ App restarted at $(date)" >> /var/log/violence-app-restarts.log
fi
EOF

chmod +x /home/admin/monitor_violence_app.sh

# Add to crontab (check every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/admin/monitor_violence_app.sh") | crontab -
```

### Backup Strategy
```bash
# Backup script
cat > /home/admin/backup_violence_app.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/admin/backups/violence_app"
mkdir -p $BACKUP_DIR

# Backup configuration
tar -czf $BACKUP_DIR/config_$(date +%Y%m%d_%H%M%S).tar.gz \
    /home/admin/web_prototype

# Keep only last 7 days
find $BACKUP_DIR -name "config_*.tar.gz" -mtime +7 -delete
EOF

chmod +x /home/admin/backup_violence_app.sh

# Run daily at 2 AM
(crontab -l 2>/dev/null; echo "0 2 * * * /home/admin/backup_violence_app.sh") | crontab -
```

### Log Rotation
```bash
# Setup log rotation
sudo tee /etc/logrotate.d/violence-detection << EOF
/var/lib/docker/containers/*/*.log {
    rotate 7
    daily
    compress
    missingok
    delaycompress
    copytruncate
}
EOF
```

## Performance Tuning

### Optimize for High Traffic
```bash
# Edit nginx config for better performance
sudo nano /etc/nginx/sites-available/violence-detection

# Add these inside server block:
# worker_connections 1024;
# keepalive_timeout 65;
# client_body_buffer_size 10M;
# client_max_body_size 100M;

# Reload nginx
sudo nginx -t && sudo systemctl reload nginx
```

### Enable Caching (Optional)
```bash
# Add caching for static files in nginx
location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
    expires 30d;
    add_header Cache-Control "public, immutable";
}
```

## Security Hardening

### Firewall Rules
```bash
# Allow only necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### Rate Limiting
```bash
# Add to nginx config
limit_req_zone $binary_remote_addr zone=videoupload:10m rate=10r/m;

location /api/predict {
    limit_req zone=videoupload burst=5;
    # ... rest of config
}
```

### Fail2ban (Optional)
```bash
sudo apt install fail2ban -y
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

## Maintenance Commands

### Update Application
```bash
cd /home/admin/web_prototype
git pull  # Or upload new files
./deploy.sh production
```

### View Logs
```bash
# Application logs
docker logs -f violence-detection-app

# Last 100 lines
docker logs --tail 100 violence-detection-app

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Restart Application
```bash
docker restart violence-detection-app
```

### Check Resource Usage
```bash
# Container stats
docker stats violence-detection-app

# System resources
htop
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs violence-detection-app

# Common issues:
# 1. Model file not found
ls -lh /home/admin/Downloads/best_model.h5

# 2. Port already in use
sudo lsof -i :8000
sudo netstat -tlnp | grep 8000

# 3. Permission issues
sudo chown -R $USER:$USER /home/admin/web_prototype
```

### SSL Certificate Issues
```bash
# Renew certificate manually
sudo certbot renew --dry-run
sudo certbot renew

# Check certificate status
sudo certbot certificates
```

### High Memory Usage
```bash
# Limit container memory
docker update --memory="4g" --memory-swap="4g" violence-detection-app

# Restart container
docker restart violence-detection-app
```

## Success Criteria

Application is successfully deployed when:
- [x] Container is running: `docker ps | grep violence-detection`
- [x] Health check passes: `curl https://vision.nexaratech.io/api/health`
- [x] Web interface loads: Browser shows upload page
- [x] Video analysis works: Upload test video, get results
- [x] SSL is active: HTTPS connection is secure
- [x] Logs show no errors: `docker logs violence-detection-app`

## Contact & Support

**Production URL**: https://vision.nexaratech.io
**Server IP**: 31.57.166.18
**Model Location**: /home/admin/Downloads/best_model.h5
**Application Dir**: /home/admin/web_prototype

---

**Status**: ✅ Ready for deployment
**Last Updated**: 2025-11-06
**Deployed By**: NexaraTech Team
