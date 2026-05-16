import pytest
import os
from fastapi.testclient import TestClient


def test_root(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "InferX"
    assert response.json()["status"] == "running"


def test_health(client):
    """Test health endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data


def test_login_success(client):
    """Test successful login"""
    import os
    response = client.post(
        "/api/auth/token",
        data={
            "username": os.getenv("TEST_USERNAME", "admin"),
            "password": os.getenv("TEST_PASSWORD", "changeme")
        }
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_login_wrong_password(client):
    """Test login with wrong password"""
    response = client.post(
        "/api/auth/token",
        data={
            "username": os.getenv("TEST_USERNAME", "admin"),
            "password": "wrongpassword"
        }
    )
    assert response.status_code == 401


def test_login_wrong_username(client):
    """Test login with wrong username"""
    response = client.post(
        "/api/auth/token",
        data={"username": "nobody", "password": "inferx123"}
    )
    assert response.status_code == 401


def test_encode_without_auth(client):
    """Test encode endpoint without authentication"""
    response = client.post(
        "/api/v1/encode",
        json={"texts": ["Hello world"], "normalize": True}
    )
    assert response.status_code == 401


def test_encode_with_auth(client, auth_headers, sample_request):
    """Test encode endpoint with valid authentication"""
    response = client.post(
        "/api/v1/encode",
        json=sample_request,
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert "shape" in data
    assert "layer_used" in data
    assert "processing_time" in data
    assert data["shape"][0] == len(sample_request["texts"])


def test_encode_response_shape(client, auth_headers):
    """Test embedding shape is correct"""
    response = client.post(
        "/api/v1/encode",
        json={"texts": ["Hello", "World", "Test"], "normalize": True},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["shape"][0] == 3


def test_encode_specific_layer(client, auth_headers):
    """Test extraction from specific layer"""
    response = client.post(
        "/api/v1/encode",
        json={"texts": ["Hello world"], "layer": 3, "normalize": True},
        headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json()["layer_used"] == 3


def test_encode_empty_texts(client, auth_headers):
    """Test encode with empty texts list"""
    response = client.post(
        "/api/v1/encode",
        json={"texts": [], "normalize": True},
        headers=auth_headers
    )
    assert response.status_code == 422


def test_get_me(client, auth_headers):
    """Test get current user endpoint"""
    import os
    response = client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["username"] == os.getenv("TEST_USERNAME", "admin")
    assert response.json()["role"] == "admin"


def test_layers_endpoint(client, auth_headers):
    """Test layers information endpoint"""
    response = client.get("/api/v1/layers", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "total_layers" in data
    assert "recommended_layers" in data