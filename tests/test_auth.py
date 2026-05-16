import pytest
from app.security.auth_backend import (
    verify_password,
    get_password_hash,
    authenticate_user,
    create_access_token,
    verify_token
)
from datetime import timedelta


def test_password_hashing():
    """Test password hashing and verification"""
    password = "testpassword123"
    hashed = get_password_hash(password)
    assert verify_password(password, hashed)
    assert not verify_password("wrongpassword", hashed)


def test_authenticate_valid_user():
    """Test authentication with valid credentials"""
    import os
    username = os.getenv("TEST_USERNAME", "admin")
    password = os.getenv("TEST_PASSWORD", "changeme")
    user = authenticate_user(username, password)
    assert user is not None
    assert user["username"] == username
    assert user["role"] == "admin"


def test_authenticate_invalid_user():
    """Test authentication with invalid credentials"""
    user = authenticate_user("nobody", "password")
    assert user is None


def test_authenticate_wrong_password():
    """Test authentication with wrong password"""
    user = authenticate_user("santanu", "wrongpassword")
    assert user is None


def test_create_and_verify_token():
    """Test JWT token creation and verification"""
    token = create_access_token(
        data={"sub": "santanu", "role": "admin"}
    )
    assert token is not None

    payload = verify_token(token)
    assert payload is not None
    assert payload["username"] == "santanu"
    assert payload["role"] == "admin"


def test_expired_token():
    """Test that expired tokens are rejected"""
    token = create_access_token(
        data={"sub": "santanu", "role": "admin"},
        expires_delta=timedelta(seconds=-1)
    )
    payload = verify_token(token)
    assert payload is None


def test_invalid_token():
    """Test that invalid tokens are rejected"""
    payload = verify_token("invalid.token.here")
    assert payload is None


def test_rate_limiting(client, auth_headers):
    """Test rate limiter allows requests within limit"""
    for _ in range(5):
        response = client.post(
            "/api/v1/encode",
            json={"texts": ["Hello"], "normalize": True},
            headers=auth_headers
        )
        assert response.status_code == 200