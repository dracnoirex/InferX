import pytest
from app.inference.layer_extractor import LayerExtractor
from app.inference.inference_engine import InferenceEngine
from app.schemas.request import EmbeddingRequest
from app.models.model_loader import model_loader


def test_embedding_shape(client, auth_headers):
    """Test embedding dimensions are correct for DistilBERT"""
    response = client.post(
        "/api/v1/encode",
        json={"texts": ["Hello world"], "normalize": True},
        headers=auth_headers
    )
    data = response.json()
    # DistilBERT hidden size = 768
    assert data["shape"][1] == 768


def test_embedding_normalized(client, auth_headers):
    """Test embeddings are unit normalized"""
    import math
    response = client.post(
        "/api/v1/encode",
        json={"texts": ["Hello world"], "normalize": True},
        headers=auth_headers
    )
    embedding = response.json()["embeddings"][0]
    norm = math.sqrt(sum(x**2 for x in embedding))
    assert abs(norm - 1.0) < 1e-5


def test_different_texts_different_embeddings(client, auth_headers):
    """Test that different texts produce different embeddings"""
    response = client.post(
        "/api/v1/encode",
        json={
            "texts": ["Hello world", "Completely different sentence"],
            "normalize": True
        },
        headers=auth_headers
    )
    embeddings = response.json()["embeddings"]
    assert embeddings[0] != embeddings[1]


def test_same_text_same_embedding(client, auth_headers):
    """Test that same text always produces same embedding"""
    request = {
        "texts": ["Hello world"],
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
    assert response1.json()["embeddings"] == response2.json()["embeddings"]


def test_batch_processing(client, auth_headers):
    """Test batch of multiple texts"""
    texts = [f"Sample text number {i}" for i in range(10)]
    response = client.post(
        "/api/v1/encode",
        json={"texts": texts, "normalize": True},
        headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json()["shape"][0] == 10


def test_layer_extraction(client, auth_headers):
    """Test extraction from different layers"""
    for layer in [0, 1, 3, 6]:
        response = client.post(
            "/api/v1/encode",
            json={"texts": ["Hello world"], "layer": layer},
            headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["layer_used"] == layer


def test_processing_time_recorded(client, auth_headers):
    """Test that processing time is always recorded"""
    response = client.post(
        "/api/v1/encode",
        json={"texts": ["Hello world"], "normalize": True},
        headers=auth_headers
    )
    assert response.json()["processing_time"] > 0