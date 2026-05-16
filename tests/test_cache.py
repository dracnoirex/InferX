import pytest
from app.cache.embedding_cache import EmbeddingCache
from app.cache.redis_client import redis_client


def test_cache_hit(client, auth_headers):
    """Test that same request returns identical embeddings"""
    from app.cache.redis_client import redis_client

    if not redis_client.is_connected:
        pytest.skip("Redis not available in test environment")

    request = {
        "texts": ["Cache test sentence unique 12345"],
        "normalize": True,
        "layer": None
    }

    response1 = client.post(
        "/api/v1/encode",
        json=request,
        headers=auth_headers
    )
    response2 = client.post(
        "/api/v1/encode",
        json=request,
        headers=auth_headers
    )

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json()["embeddings"] == response2.json()["embeddings"]

def test_cache_key_uniqueness():
    """Test that different inputs produce different cache keys"""
    cache = EmbeddingCache()
    key1 = cache._make_key(["Hello"], None, True)
    key2 = cache._make_key(["World"], None, True)
    key3 = cache._make_key(["Hello"], 3, True)
    assert key1 != key2
    assert key1 != key3


def test_cache_stores_and_retrieves():
    """Test cache set and get operations"""
    if not redis_client.is_connected:
        pytest.skip("Redis not available")

    cache = EmbeddingCache()
    texts = ["Test cache storage unique xyz"]
    result = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "shape": [1, 3],
        "layer_used": 6,
        "total_layers": 6
    }

    cache.set(texts, None, True, result)
    retrieved = cache.get(texts, None, True)

    assert retrieved is not None
    assert retrieved["embeddings"] == result["embeddings"]