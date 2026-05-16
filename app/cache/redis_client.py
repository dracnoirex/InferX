import redis
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class RedisClient:
    def __init__(self):
        self.client = None
        self.is_connected = False

    def connect(self):
        """Connect to Redis server"""
        try:
            self.client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True
            )
            self.client.ping()
            self.is_connected = True
            logger.info("✅ Redis connected!")

        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}")
            self.is_connected = False

    def get(self, key: str):
        """Get value from cache"""
        if not self.is_connected:
            return None
        try:
            return self.client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: str, ttl: int = 3600):
        """Set value in cache with TTL (default 1 hour)"""
        if not self.is_connected:
            return False
        try:
            self.client.setex(key, ttl, value)
            return True
        except Exception:
            return False

    def delete(self, key: str):
        """Delete value from cache"""
        if not self.is_connected:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception:
            return False


# Global redis client
redis_client = RedisClient()