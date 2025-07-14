# Fly.io Migration Guide

## Overview
This guide covers the migration from Render to Fly.io for cost optimization. The migration focuses on the backend Python FastAPI server while keeping the frontend on Vercel.

## Cost Comparison

| Service | Render | Fly.io |
|---------|--------|--------|
| **Backend** | $7/month (Starter) | $0/month (Free tier: 3 shared-cpu-1x, 160GB/month) |
| **Frontend** | N/A (using Vercel) | $0/month (or stay on Vercel free) |
| **Total Savings** | **$84/year** | **$0/year** |

## Migration Files Created

### Backend Configuration
- `backend/fly.toml` - Fly.io app configuration
- `backend/Dockerfile` - Custom Docker setup for ML dependencies
- `backend/.dockerignore` - Optimize Docker builds
- `backend/main.py` - Updated with health check and dynamic port configuration

### Frontend Configuration (Optional)
- `project/fly.toml` - Alternative to Vercel
- `project/Dockerfile` - Multi-stage build setup
- `project/nginx.conf` - SPA routing configuration

## Risk Assessment

### 🔴 HIGH RISK: Heavy Dependencies
- **Problem**: `insightface` + `opencv` + `onnxruntime` are heavy ML libraries
- **Impact**: Docker build takes 5-10 minutes, container size 1GB+
- **Solution**: 
  - Fly.io build time limit (15 minutes) is sufficient
  - Only initial deployment takes time

### 🟡 MEDIUM RISK: Memory Usage
- **Problem**: Face recognition model memory usage
- **Impact**: 1GB RAM might be insufficient
- **Solution**: 
  - Configured for 1GB RAM initially
  - Can scale to 2GB if needed

### 🟡 MEDIUM RISK: Cold Start
- **Problem**: Model loading takes 30-60 seconds
- **Impact**: First request might timeout
- **Solution**: 
  - Health check grace period configured
  - Auto-scaling can keep minimum 1 instance

## Migration Steps

### Step 1: Install Fly.io CLI
```bash
# macOS
brew install flyctl

# Linux/Windows
curl -L https://fly.io/install.sh | sh
```

### Step 2: Deploy Backend to Fly.io
```bash
cd backend
fly auth login
fly launch --copy-config
# Choose: Yes to copy configuration from fly.toml
# Choose: No to Postgres database (not needed)
fly deploy
```

### Step 3: Update Frontend API Endpoint
Update your frontend's API base URL to point to the new Fly.io backend:
- Old: `https://your-app-name.onrender.com`
- New: `https://face-rating-backend.fly.dev`

### Step 4: Test & Verify
```bash
# Test health endpoint
curl https://face-rating-backend.fly.dev/health

# Test main API endpoint
curl https://face-rating-backend.fly.dev/
```

## Configuration Details

### Backend Optimizations
- ✅ Auto-scaling to 0 machines (saves money when not in use)
- ✅ Tokyo region (nrt) for better Japan latency
- ✅ Shared CPU for cost optimization
- ✅ 1GB RAM (sufficient for ML workload)
- ✅ Health checks for reliability
- ✅ Optimized Dockerfile for ML dependencies

### Key Benefits
- 🎯 **Free tier**: Up to 3 shared-cpu machines, 160GB bandwidth/month
- 🔄 **Auto-sleep**: Machines sleep when not in use (saves money)
- 🚀 **Fast cold starts**: Usually under 1 second
- 🌏 **Better latency**: Tokyo region vs US-based Render

## Safe Migration Strategy

### Phase 1: Test Environment Validation
```bash
# 1. Create Fly.io app
cd backend
fly launch --copy-config

# 2. Test deployment
fly deploy

# 3. Health check
curl https://face-rating-backend.fly.dev/health
```

### Phase 2: Frontend Connection Test
```bash
# Set Vercel environment variable temporarily to new URL
VITE_API_URL=https://face-rating-backend.fly.dev
```

### Phase 3: Production Migration
- Keep Render running while testing Fly.io
- Switch DNS/environment variables gradually
- Monitor for issues before shutting down Render

## Monitoring & Troubleshooting

### Monitoring Commands
```bash
# Check app status
fly status

# View logs
fly logs

# Check resource usage
fly machine status
```

### Common Issues and Solutions

1. **Memory Issues**: Scale up to 2GB RAM if needed
2. **Cold Start Timeout**: Configure health check grace period
3. **Build Timeout**: Optimize Dockerfile, use multi-stage builds
4. **Database Connection**: Update connection strings if needed

## Rollback Plan

If issues occur:
1. Revert frontend environment variables to Render URLs
2. Keep Render backend running until Fly.io is stable
3. DNS changes can be reverted quickly

## Additional Resources

- [Fly.io Documentation](https://fly.io/docs/)
- [Fly.io Python Guide](https://fly.io/docs/languages-and-frameworks/python/)
- [Fly.io Pricing](https://fly.io/docs/about/pricing/)

## Conclusion

The migration to Fly.io offers significant cost savings ($84/year) while providing better performance for Japanese users. The main risks are related to heavy ML dependencies, but these can be mitigated with proper configuration and monitoring.

The gradual migration approach ensures minimal downtime and provides a rollback option if issues occur.