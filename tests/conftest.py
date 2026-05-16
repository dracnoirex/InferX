import pytest
import os
from fastapi.testclient import TestClient
from app.main import app

TEST_USERNAME = os.getenv("TEST_USERNAME", "admin")
TEST_PASSWORD = os.getenv("TEST_PASSWORD", "changeme")

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session")
def auth_token(client):
    response = client.post(
        "/api/auth/token",
        data={"username": TEST_USERNAME, "password": TEST_PASSWORD}
    )
    return response.json()["access_token"]

@pytest.fixture(scope="session")
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}

@pytest.fixture(scope="function")
def sample_texts():
    return ["Hello world", "Machine learning is amazing"]

@pytest.fixture(scope="function")
def sample_request():
    return {
        "texts": ["Hello world", "Machine learning"],
        "layer": None,
        "normalize": True
    }