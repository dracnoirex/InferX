import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.models.model_loader import model_loader
from app.monitoring.logging_config import setup_logging
from app.api import routes, auth
from prometheus_client import make_asgi_app
import logging

setup_logging()
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🟢 Startup
    logger.info("🚀 InferX starting up...")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Device: {settings.device}")

    # Initialize database
    from app.db.database import init_db
    from app.db.database import SessionLocal
    from app.db.crud import register_model
    init_db()

    # Register model in database
    db = SessionLocal()
    try:
        register_model(db, settings.model_name, settings.app_version)
    finally:
        db.close()

    # Connect Redis
    from app.cache.redis_client import redis_client
    redis_client.connect()

    # Load model
    model_loader.load()
    logger.info("✅ InferX is ready!")

    yield

    # 🔴 Shutdown
    logger.info("⏹️ InferX shutting down...")
    model_loader.unload()
    logger.info("✅ Cleanup complete")


# ── FastAPI App ───────────────────────────────────────
app = FastAPI(
    title="InferX",
    description="""
    ## Production-grade ML Inference API

    Extract custom hidden states and embeddings from LLMs.

    ### Features
    - 🧠 Hidden layer extraction
    - ⚡ GPU-accelerated inference
    - 🔐 JWT Authentication
    - 🚦 Rate limiting
    - 📊 Prometheus metrics
    """,
    version=settings.app_version,
    lifespan=lifespan
)

# ── Middleware ────────────────────────────────────────
# Read allowed origins from environment
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,    # specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Routes ────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(routes.router)

# ── Prometheus metrics endpoint ───────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ── Root ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "InferX",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics"
    }


# ── Run ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )