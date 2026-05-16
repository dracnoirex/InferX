import json
import hashlib
from typing import Optional, List
from app.cache.redis_client import redis_client
from app.monitoring.metrics import CACHE_HIT_COUNT, CACHE_MISS_COUNT
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl  

    def _make_key(
        self,
        texts: List[str],
        layer: Optional[int],
        normalize: bool
    ) -> str:
       
        raw = f"{sorted(texts)}:{layer}:{normalize}"

        hash_key = hashlib.md5(raw.encode()).hexdigest()

        return f"inferx:embedding:{hash_key}"

    def get(
        self,
        texts: List[str],
        layer: Optional[int],
        normalize: bool
    ) -> Optional[dict]:
        key = self._make_key(texts, layer, normalize)
        cached = redis_client.get(key)

        if cached:
            CACHE_HIT_COUNT.inc()
            logger.info(f"🎯 Cache HIT! key={key[:20]}...")
            return json.loads(cached)

        CACHE_MISS_COUNT.inc()
        logger.info(f"❌ Cache MISS! key={key[:20]}...")
        return None

    def set(
        self,
        texts: List[str],
        layer: Optional[int],
        normalize: bool,
        result: dict
    ) -> bool:
        key = self._make_key(texts, layer, normalize)
        value = json.dumps(result)
        success = redis_client.set(key, value, ttl=self.ttl)

        if success:
            logger.info(f"💾 Cached! key={key[:20]}...")

        return success

    def invalidate_all(self) -> bool:
     
        try:
            keys = redis_client.client.keys("inferx:embedding:*")
            if keys:
                redis_client.client.delete(*keys)
                logger.info(f"🗑️ Cleared {len(keys)} cached embeddings")
            return True
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return False


# Global cache object
embedding_cache = EmbeddingCache(ttl=3600)