from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.models import RequestLog, User, ModelVersion
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ── Request Logs ──────────────────────────────────────

def log_request(
    db: Session,
    username: str,
    endpoint: str,
    num_texts: int,
    layer_used: int,
    processing_time: float,
    cache_hit: bool,
    model_name: str,
    status_code: int
) -> RequestLog:
    """Log an API request to database"""
    log = RequestLog(
        username=username,
        endpoint=endpoint,
        num_texts=num_texts,
        layer_used=layer_used,
        processing_time=processing_time,
        cache_hit=cache_hit,
        model_name=model_name,
        status_code=status_code
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_request_stats(db: Session, username: str = None) -> dict:
    """Get request statistics"""
    query = db.query(RequestLog)

    if username:
        query = query.filter(RequestLog.username == username)

    total = query.count()
    cache_hits = query.filter(RequestLog.cache_hit == True).count()
    avg_latency = db.query(
        func.avg(RequestLog.processing_time)
    ).scalar() or 0.0

    return {
        "total_requests": total,
        "cache_hits": cache_hits,
        "cache_hit_rate": f"{(cache_hits/total*100):.1f}%" if total > 0 else "0%",
        "avg_latency_seconds": round(avg_latency, 4)
    }


def get_recent_requests(
    db: Session,
    limit: int = 10,
    username: str = None
) -> list:
    """Get recent requests"""
    query = db.query(RequestLog)
    if username:
        query = query.filter(RequestLog.username == username)
    return query.order_by(
        RequestLog.created_at.desc()
    ).limit(limit).all()


# ── Users ─────────────────────────────────────────────

def get_or_create_user(db: Session, username: str, role: str = "user") -> User:
    """Get existing user or create new one"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(username=username, role=role)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def update_user_activity(db: Session, username: str) -> None:
    """Update user last seen and request count"""
    user = db.query(User).filter(User.username == username).first()
    if user:
        user.total_requests += 1
        user.last_seen = datetime.utcnow()
        db.commit()


def get_user_stats(db: Session, username: str) -> dict:
    """Get stats for a specific user"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {}

    request_stats = get_request_stats(db, username)

    return {
        "username": user.username,
        "role": user.role,
        "total_requests": user.total_requests,
        "member_since": user.created_at.isoformat(),
        "last_seen": user.last_seen.isoformat(),
        **request_stats
    }


# ── Model Versions ────────────────────────────────────

def register_model(
    db: Session,
    model_name: str,
    version: str = "1.0.0"
) -> ModelVersion:
    """Register a model version"""
    # Deactivate previous versions
    db.query(ModelVersion).filter(
        ModelVersion.model_name == model_name
    ).update({"is_active": False})

    model = ModelVersion(
        model_name=model_name,
        version=version,
        is_active=True
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    logger.info(f"✅ Model registered: {model_name} v{version}")
    return model


def get_active_model(db: Session, model_name: str) -> ModelVersion:
    """Get currently active model version"""
    return db.query(ModelVersion).filter(
        ModelVersion.model_name == model_name,
        ModelVersion.is_active == True
    ).first()


def update_model_stats(
    db: Session,
    model_name: str,
    latency: float
) -> None:
    """Update model request count and average latency"""
    model = get_active_model(db, model_name)
    if model:
        model.total_requests += 1
        # Rolling average
        model.avg_latency = (
            (model.avg_latency * (model.total_requests - 1) + latency)
            / model.total_requests
        )
        db.commit()