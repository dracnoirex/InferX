from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.sql import func
from app.db.database import Base


class RequestLog(Base):
    """Log every API request"""
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    endpoint = Column(String)
    num_texts = Column(Integer)
    layer_used = Column(Integer)
    processing_time = Column(Float)
    cache_hit = Column(Boolean, default=False)
    model_name = Column(String)
    status_code = Column(Integer)
    created_at = Column(DateTime, default=func.now())


class User(Base):
    """User management"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="user")
    is_active = Column(Boolean, default=True)
    total_requests = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    last_seen = Column(DateTime, default=func.now())


class ModelVersion(Base):
    """Track model versions"""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    version = Column(String)
    is_active = Column(Boolean, default=True)
    total_requests = Column(Integer, default=0)
    avg_latency = Column(Float, default=0.0)
    deployed_at = Column(DateTime, default=func.now())