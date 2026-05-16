#!/bin/bash
# InferX Deployment Script

set -e

echo "🚀 Deploying InferX..."

# Build Docker image
echo "📦 Building Docker image..."
docker build -f docker/Dockerfile -t inferx:latest .

# Stop existing containers
echo "⏹️ Stopping existing containers..."
docker stop inferx-api inferx-redis 2>/dev/null || true
docker rm inferx-api inferx-redis 2>/dev/null || true

# Start Redis
echo "🔴 Starting Redis..."
docker run -d \
  --name inferx-redis \
  -p 6379:6379 \
  --restart unless-stopped \
  redis:alpine

# Start InferX
echo "⚡ Starting InferX..."
docker run -d \
  --name inferx-api \
  --network host \
  -e REDIS_HOST=localhost \
  -e DEVICE=cpu \
  -p 8000:8000 \
  --restart unless-stopped \
  inferx:latest

# Health check
echo "🔍 Running health check..."
sleep 10
python scripts/healthcheck.py

echo "✅ InferX deployed successfully!"