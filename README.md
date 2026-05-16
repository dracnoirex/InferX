<div align="center">

# ⚡ InferX

**Production-grade · GPU-Accelerated · ML Inference API**
Open-source hidden state and embedding extraction from transformer models

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-30%20passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](docker/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

<br/>

> *"Extract embeddings from any transformer layer — with GPU acceleration, intelligent caching, and enterprise-grade security."*

</div>

---

## 📋 Table of Contents

- [What is InferX?](#-what-is-inferx)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Python SDK Usage](#-python-sdk-usage)
- [Layer Guide](#-layer-guide)
- [Supported Models](#-supported-models)
- [Monitoring](#-monitoring)
- [Security](#-security)
- [Docker Deployment](#-docker-deployment)
- [Running Tests](#-running-tests)
- [Benchmarking](#-benchmarking)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)

---

## 🌍 What is InferX?

InferX is a **production-ready ML inference API** built in Python with FastAPI. It solves a problem that standard LLM APIs ignore entirely — **access to internal model representations**.

| Tool | Problem |
|------|---------|
| **OpenAI API** | Final output only — no hidden states, no layer control |
| **HuggingFace Inference API** | Limited customization, no layer selection |
| **TorchServe / Triton** | Generic serving — not built for embedding extraction |

InferX fills this gap — giving you fine-grained control over which layer to extract, with Redis caching for 1000x speedup, JWT authentication, rate limiting, database analytics, and full observability.

---

## ✨ Features

| Feature | What it does |
|---------|-------------|
| **Hidden Layer Extraction** | Extract embeddings from any transformer layer — not just the last one |
| **GPU Acceleration** | CUDA-powered inference on NVIDIA GPUs with automatic CPU fallback |
| **Redis Caching** | 1000x faster responses — 4600ms cold inference → 4ms cache hit |
| **JWT Authentication** | OAuth2 + JWT with access tokens (30min) and refresh tokens (7 days) |
| **Rate Limiting** | Sliding window per-user limits — 60 req/min, 1000 req/day |
| **Database Analytics** | SQLite request logging, user stats, model registry, rolling averages |
| **Batch Processing** | Dynamic batching with timeout protection and size validation |
| **Background Jobs** | Celery + Redis for async large-batch embedding tasks |
| **Prometheus Metrics** | P50/P95/P99 latency, cache hit rate, GPU memory, throughput |
| **Structured Logging** | JSON-formatted logs with module, function, line, and timestamp |
| **Model Registry** | Track supported models, versions, VRAM requirements |
| **Docker Ready** | Multi-stage Dockerfile + full docker-compose stack |
| **30/30 Tests** | Comprehensive test suite with CI/CD via GitHub Actions |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Client Layer                        │
│              REST API  ·  Swagger UI  ·  SDK            │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              JWT Auth Middleware                        │
│      OAuth2PasswordBearer · token verify · role check   │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Rate Limiter                               │
│       60 req/min · 1000 req/day · per-user sliding window│
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Request Validation                         │
│       Pydantic · max 100 texts · max 10,000 chars each  │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Redis Cache Check                          │
│       MD5 key · TTL 1hr · hit → return instantly ⚡     │
└──────────┬──────────────────────────────────────────────┘
           │ MISS
           ↓
┌─────────────────────────────────────────────────────────┐
│              Inference Engine                           │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Layer Extractor                      │  │
│  │  Tokenize → GPU Inference → Hidden State Select  │  │
│  │  → Mean Pooling → Normalize → Return Embedding   │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│         Cache Save · DB Log · Metrics Update            │
│    Redis TTL=1hr · SQLite log · Prometheus counters     │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Return Response                        │
│     embeddings · shape · layer_used · processing_time   │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git
- Docker Desktop (for Redis)
- NVIDIA GPU with CUDA 12.0+ (optional — CPU supported)

### Option 1 — Local Setup

**Step 1 — Clone & setup environment**

```bash
git clone https://github.com/batman-512/InferX.git
cd InferX

conda create -n inferx python=3.10 -y
conda activate inferx
```

**Step 2 — Install PyTorch**

```bash
# GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio
```

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
pip install sqlalchemy alembic aiosqlite
```

**Step 4 — Configure environment**

```bash
cp .env.example .env
# Edit .env — set ADMIN_USERNAME, ADMIN_PASSWORD, SECRET_KEY
```

**Step 5 — Start Redis**

```bash
docker run -d --name inferx-redis -p 6379:6379 redis:alpine
```

**Step 6 — Start InferX**

```bash
python -m app.main
```

Output:
```
✅ Logging setup complete
✅ Database initialized
✅ Redis connected!
Loading model: distilbert-base-uncased
✅ Model loaded successfully on cuda
GPU Memory used: 254.2 MB
✅ InferX is ready!
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Step 7 — Open Swagger UI**

```
http://localhost:8000/docs
```

---

### Option 2 — Docker Setup

```bash
git clone https://github.com/batman-512/InferX.git
cd InferX
cp .env.example .env

# Build image
docker build -f docker/Dockerfile -t inferx:latest .

# Start Redis
docker run -d --name inferx-redis -p 6379:6379 redis:alpine

# Run InferX
docker run -d \
  --name inferx-api \
  --network host \
  -e DEVICE=cpu \
  -p 8000:8000 \
  inferx:latest
```

---

### Option 3 — Full Stack with docker-compose

```bash
git clone https://github.com/batman-512/InferX.git
cd InferX
cp .env.example .env

docker-compose -f docker/docker-compose.yaml up -d
```

Starts: InferX API · Redis · Prometheus · Grafana

---

## 🔧 Configuration

All configuration via `.env`. Copy `.env.example` and edit:

```env
# ── App ───────────────────────────────────────────────
APP_NAME=InferX
APP_VERSION=1.0.0
DEBUG=false
HOST=0.0.0.0
PORT=8000

# ── Model ─────────────────────────────────────────────
MODEL_NAME=distilbert-base-uncased
MAX_LENGTH=512
BATCH_SIZE=8
DEVICE=cuda          # cuda | cpu

# ── Security — MUST change in production! ─────────────
SECRET_KEY=change-this-to-a-random-secret-minimum-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# ── Admin credentials — set via env, never hardcode ───
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change-this-strong-password

# ── Redis ─────────────────────────────────────────────
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# ── CORS ──────────────────────────────────────────────
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# ── Grafana ───────────────────────────────────────────
GRAFANA_PASSWORD=change-this-grafana-password

# ── HuggingFace (optional — for private models) ───────
HF_TOKEN=your-huggingface-token-here
```

> ⚠️ **Never commit `.env` to Git.** It is already in `.gitignore`.

---

## 📡 API Reference

### Authentication

#### `POST /api/auth/token` — Login

**Request:**
```bash
curl -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=yourpassword"
```

**Response:**
```json
{
  "access_token": "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type": "bearer",
  "username": "admin",
  "role": "admin",
  "expires_in": 1800
}
```

#### `POST /api/auth/refresh` — Refresh token

```bash
curl -X POST http://localhost:8000/api/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "your_refresh_token"}'
```

Returns new access token + new refresh token. No re-login needed.

#### `GET /api/auth/me` — Current user info

```bash
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer <access_token>"
```

```json
{"username": "admin", "role": "admin"}
```

---

### Inference

#### `POST /api/v1/encode` — Extract embeddings

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/encode \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "Machine learning is amazing"],
    "layer": null,
    "normalize": true
  }'
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `texts` | `List[str]` | required | Texts to encode (max 100) |
| `layer` | `int \| null` | `null` | Hidden layer to extract (`null` = last layer) |
| `normalize` | `bool` | `true` | Normalize embeddings to unit length |

**Response:**
```json
{
  "embeddings": [[0.042, -0.014, -0.014, ...]],
  "shape": [2, 768],
  "layer_used": 6,
  "model_name": "distilbert-base-uncased",
  "processing_time": 0.004
}
```

| Field | Description |
|-------|-------------|
| `embeddings` | List of embedding vectors — one per input text |
| `shape` | `[batch_size, hidden_size]` |
| `layer_used` | Which layer was extracted |
| `processing_time` | Seconds — `< 0.05` means cache hit |

---

#### `GET /api/v1/health` — Health check

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_memory_used": "254.2 MB",
  "gpu_memory_total": "4096.0 MB",
  "model_name": "distilbert-base-uncased",
  "version": "1.0.0"
}
```

---

#### `GET /api/v1/layers` — Layer information

```bash
curl http://localhost:8000/api/v1/layers \
  -H "Authorization: Bearer <access_token>"
```

```json
{
  "model_name": "distilbert-base-uncased",
  "total_layers": 6,
  "recommended_layers": {
    "surface": [0, 1],
    "middle": [2, 3],
    "deep": [4, 5, 6]
  }
}
```

---

#### `GET /api/v1/stats` — Usage statistics (admin)

```bash
curl http://localhost:8000/api/v1/stats \
  -H "Authorization: Bearer <access_token>"
```

```json
{
  "user": "admin",
  "role": "admin",
  "total_requests": 1500,
  "cache_hits": 1380,
  "cache_hit_rate": "92.0%",
  "avg_latency_seconds": 0.187
}
```

---

#### `GET /api/v1/me/stats` — Personal statistics

```bash
curl http://localhost:8000/api/v1/me/stats \
  -H "Authorization: Bearer <access_token>"
```

```json
{
  "username": "admin",
  "role": "admin",
  "total_requests": 42,
  "member_since": "2026-04-24T00:00:00",
  "last_seen": "2026-04-24T20:00:00",
  "cache_hits": 38,
  "cache_hit_rate": "90.5%",
  "avg_latency_seconds": 0.012
}
```

---

#### `GET /metrics` — Prometheus metrics scrape

```bash
curl http://localhost:8000/metrics
```

Returns raw Prometheus metrics including:
- `inferx_request_total` — total requests by endpoint + status
- `inferx_inference_latency_seconds` — latency histogram
- `inferx_cache_hit_total` / `inferx_cache_miss_total`
- `inferx_gpu_memory_used_mb` / `inferx_gpu_memory_total_mb`

---

## 🐍 Python SDK Usage

```python
import httpx

BASE_URL = "http://localhost:8000"

# Step 1 — Authenticate
response = httpx.post(
    f"{BASE_URL}/api/auth/token",
    data={"username": "admin", "password": "yourpassword"}
)
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Step 2 — Extract embeddings
response = httpx.post(
    f"{BASE_URL}/api/v1/encode",
    json={
        "texts": ["Hello world", "Machine learning"],
        "layer": None,      # None = last layer (recommended)
        "normalize": True
    },
    headers=headers
)

data = response.json()
print(f"Shape:   {data['shape']}")            # [2, 768]
print(f"Layer:   {data['layer_used']}")       # 6
print(f"Time:    {data['processing_time']}s") # 0.004s (cache hit)
embeddings = data["embeddings"]               # List[List[float]]

# Step 3 — Compute similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

emb = np.array(embeddings)
similarity = cosine_similarity([emb[0]], [emb[1]])[0][0]
print(f"Similarity: {similarity:.4f}")        # e.g. 0.8734
```

---

## 🔬 Layer Guide

InferX lets you extract embeddings from any hidden layer. Different layers capture different types of information:

| Layer Range | What it captures | Best for |
|-------------|-----------------|----------|
| `0–1` | Surface features — raw word patterns | Spell checking, morphology |
| `2–3` | Syntactic features — grammar structure | POS tagging, parsing |
| `4–6` | Semantic features — sentence meaning | Semantic search, similarity |
| `null` (last) | Highest abstraction | General purpose (recommended) |

**Example — extract from specific layer:**
```bash
curl -X POST http://localhost:8000/api/v1/encode \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"], "layer": 3, "normalize": true}'
```

---

## 🔄 Supported Models

Change `MODEL_NAME` in `.env` to use any HuggingFace transformer:

| Model | Params | VRAM | Best for |
|-------|--------|------|---------|
| `distilbert-base-uncased` | 66M | ~300MB | Fast inference, general use |
| `bert-base-uncased` | 110M | ~500MB | Higher accuracy NLP |
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | ~100MB | Sentence similarity |
| `microsoft/phi-2` | 2.7B | ~3.5GB | Advanced language tasks |

> Any HuggingFace AutoModel-compatible model works. Set `MODEL_NAME` and restart.

---

## 📊 Monitoring

| Service | URL | Credentials |
|---------|-----|-------------|
| Swagger UI | http://localhost:8000/docs | — |
| Prometheus Metrics | http://localhost:8000/metrics | — |
| Prometheus Dashboard | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / from `.env` |

> ⚠️ Make sure `.env` has `GRAFANA_PASSWORD` set before running docker-compose.

---

## 🔒 Security

### JWT Authentication

InferX uses OAuth2 + JWT with two-token system:

- **Access token** — expires in 30 minutes (configurable)
- **Refresh token** — expires in 7 days, used to get new access token

```bash
# Login — get both tokens
curl -X POST http://localhost:8000/api/auth/token \
  -d "username=admin&password=yourpassword"

# Refresh — get new access token without re-login
curl -X POST http://localhost:8000/api/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "your_refresh_token"}'
```

### Rate Limiting

Per-user sliding window limits:

| Limit | Value | Behavior |
|-------|-------|----------|
| Per minute | 60 requests | HTTP 429 after exceeded |
| Per day | 1000 requests | HTTP 429 after exceeded |

### Input Validation

- Max 100 texts per request — HTTP 400 if exceeded
- Max 10,000 characters per text — HTTP 400 if exceeded
- No stack traces exposed to clients in error responses

### CORS

Configure allowed origins in `.env`:

```env
# Development
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# Production
ALLOWED_ORIGINS=https://your-app.com
```

> `allow_origins=["*"]` with `allow_credentials=True` is intentionally blocked — modern browsers reject this combination. InferX uses explicit origin whitelisting.

---

## 🐳 Docker Deployment

### Server Only

```bash
# Build
docker build -f docker/Dockerfile -t inferx:latest .

# Run
docker run -d \
  --name inferx-api \
  --network host \
  --env-file .env \
  -p 8000:8000 \
  inferx:latest
```

### Full Stack

```bash
docker-compose -f docker/docker-compose.yaml up -d
```

| Service | URL |
|---------|-----|
| InferX API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### Verify Deployment

```bash
python scripts/healthcheck.py
```

Output:
```
✅ Root endpoint
✅ Health endpoint — model=distilbert-base-uncased, device=cuda
✅ Authentication
✅ Inference endpoint — shape=[1, 768], time=0.004s
✅ Prometheus metrics

✅ All checks passed!
```

---

## 🧪 Running Tests

```bash
# All 30 tests
python -m pytest tests/ -v

# By module
python -m pytest tests/test_api.py -v        # 12 tests — endpoints + auth flow
python -m pytest tests/test_auth.py -v       # 8 tests  — JWT, tokens, hashing
python -m pytest tests/test_cache.py -v      # 3 tests  — cache hit/miss/keys
python -m pytest tests/test_inference.py -v  # 7 tests  — shape, norm, layers
```

Expected:
```
30 passed, 0 errors ✅
```

### With Redis (all 30 tests pass)

```bash
docker run -d --name inferx-redis -p 6379:6379 redis:alpine
python -m pytest tests/ -v
```

### Without Redis (28 pass, 2 skip)

```bash
python -m pytest tests/ -v
# 28 passed, 2 skipped (Redis not available) ✅
```

---

## 📈 Benchmarking

### Built-in Stats

After sending some requests:

```bash
curl http://localhost:8000/api/v1/stats \
  -H "Authorization: Bearer <token>"
```

```json
{
  "total_requests": 1500,
  "cache_hits": 1380,
  "cache_hit_rate": "92.0%",
  "avg_latency_seconds": 0.005
}
```

### Load Testing Script

```bash
# Terminal 1 — start server
python -m app.main

# Terminal 2 — run load test
python scripts/load_test.py
```

Output:
```
🚀 InferX Load Test
   Requests: 20
   Concurrency: 5
   Target: http://localhost:8000

✅ Authentication successful

Progress: 5/20
Progress: 10/20
Progress: 15/20
Progress: 20/20

📊 Results:
   Total requests : 20
   Successful     : 20
   Failed         : 0
   Min latency    : 0.003s
   Max latency    : 4.821s
   Avg latency    : 0.987s
   Median latency : 0.004s
   Throughput     : 1.24 req/s
```

### Performance Benchmarks

| Scenario | Latency |
|----------|---------|
| Cache HIT | ~4ms |
| Cache MISS (GPU) | ~4600ms |
| Cache MISS (CPU) | ~8000ms |
| After warmup | ~4ms (90%+ hit rate) |

---

## 📁 Project Structure

```
inferx/
│
├── app/
│   ├── api/
│   │   ├── auth.py              # Login, refresh, /me endpoints
│   │   ├── routes.py            # /encode, /health, /layers, /stats
│   │   └── dependencies.py      # Auth + rate limit dependency chain
│   │
│   ├── cache/
│   │   ├── redis_client.py      # Redis connection manager
│   │   └── embedding_cache.py   # MD5-keyed embedding cache
│   │
│   ├── db/
│   │   ├── database.py          # SQLAlchemy engine + session
│   │   ├── models.py            # request_logs, users, model_versions tables
│   │   └── crud.py              # log_request, get_stats, update_user_activity
│   │
│   ├── inference/
│   │   ├── inference_engine.py  # Cache check → model → cache save pipeline
│   │   ├── layer_extractor.py   # Tokenize → GPU → hidden states → mean pool
│   │   └── batch_processor.py   # Batch splitting with timeout protection
│   │
│   ├── middleware/
│   │   ├── request_id.py        # Unique X-Request-ID on every response
│   │   ├── cors.py              # CORS reference (applied in main.py)
│   │   └── compression.py       # GZip response compression
│   │
│   ├── models/
│   │   ├── model_loader.py      # Model lifecycle — load, unload, GPU placement
│   │   └── model_registry.py    # Supported models + VRAM requirements
│   │
│   ├── monitoring/
│   │   ├── metrics.py           # Prometheus Counter, Histogram, Gauge + helpers
│   │   └── logging_config.py    # JSON structured logging setup
│   │
│   ├── schemas/
│   │   ├── request.py           # EmbeddingRequest — Pydantic validation
│   │   └── response.py          # EmbeddingResponse, HealthResponse, ErrorResponse
│   │
│   ├── security/
│   │   ├── auth_backend.py      # bcrypt hashing, JWT create/verify, env-based users
│   │   └── rate_limiter.py      # Sliding window rate limiter per user
│   │
│   ├── tasks/
│   │   ├── celery_app.py        # Celery + Redis broker setup
│   │   └── background_jobs.py   # Async batch encode task + log cleanup
│   │
│   ├── config.py                # Pydantic Settings — .env loading
│   └── main.py                  # FastAPI app — lifespan, middleware, routes, metrics
│
├── configs/
│   ├── config.yaml              # App configuration template
│   └── prometheus.yml           # Prometheus scrape config
│
├── docker/
│   ├── Dockerfile               # Multi-stage build — slim production image
│   └── docker-compose.yaml      # inferx + redis + prometheus + grafana
│
├── docs/
│   └── architecture.md          # Detailed system architecture
│
├── scripts/
│   ├── healthcheck.py           # Post-deploy verification script
│   ├── load_test.py             # Async concurrent load tester
│   └── deploy.sh                # One-command Docker deployment
│
├── tests/
│   ├── conftest.py              # Shared fixtures — client, auth_token, headers
│   ├── test_api.py              # 12 tests — all endpoints, auth flow
│   ├── test_auth.py             # 8 tests  — JWT, hashing, rate limits
│   ├── test_cache.py            # 3 tests  — hit/miss/key uniqueness
│   └── test_inference.py        # 7 tests  — shape, norm, layers, batching
│
├── .github/
│   └── workflows/
│       └── test.yaml            # CI — install, Redis service, pytest
│
├── streamlit_app.py             # Interactive demo — embeddings, similarity, layers
├── main.py                      # Alias entry point
├── requirements.txt             # Pinned dependencies
├── .env.example                 # Environment template (copy to .env)
├── .gitignore                   # Ignores .env, *.db, __pycache__, models/
└── README.md
```

---

## 🗺️ Roadmap

- [ ] gRPC support alongside REST
- [ ] Multi-model serving — load multiple models simultaneously
- [ ] Kubernetes Helm chart with HPA
- [ ] A/B testing framework for model comparison
- [ ] Token refresh rotation (one-time use refresh tokens)
- [ ] PostgreSQL support for production-scale analytics
- [ ] WebSocket streaming for real-time embedding generation
- [ ] Bengali language model fine-tuning demo
- [ ] OpenTelemetry distributed tracing (Jaeger)
- [ ] Per-user Grafana dashboards

---

## 🤝 Built With

| Technology | Role |
|-----------|------|
| [FastAPI](https://fastapi.tiangolo.com) | Async HTTP server + Swagger UI |
| [PyTorch](https://pytorch.org) | GPU inference engine |
| [HuggingFace Transformers](https://huggingface.co/docs/transformers) | Model loading + hidden state extraction |
| [Redis](https://redis.io) | Embedding cache + Celery broker |
| [SQLAlchemy](https://sqlalchemy.org) | ORM + database analytics |
| [Pydantic v2](https://docs.pydantic.dev) | Request/response validation |
| [python-jose](https://github.com/mpdavis/python-jose) | JWT token encoding/decoding |
| [passlib](https://passlib.readthedocs.io) | bcrypt password hashing |
| [Prometheus](https://prometheus.io) | Metrics collection |
| [Grafana](https://grafana.com) | Metrics visualization |
| [Celery](https://docs.celeryq.dev) | Background job processing |
| [Streamlit](https://streamlit.io) | Interactive demo UI |
| [Docker](https://docker.com) | Containerization |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built from scratch — a production-grade ML infrastructure portfolio project.**

*If this helped you, please give it a ⭐*

[![GitHub stars](https://img.shields.io/github/stars/dracnoirex/InferX?style=social)](https://github.com/dracnoirex/InferX/stargazers)

</div>