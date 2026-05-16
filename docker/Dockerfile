# ── Stage 1: Base image ───────────────────────────────
FROM python:3.10-slim as base

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# ── Stage 2: Dependencies ─────────────────────────────
FROM base as dependencies

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Production ───────────────────────────────
FROM dependencies as production

COPY app/ ./app/
COPY configs/ ./configs/
COPY .env .

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["python", "-m", "app.main"]