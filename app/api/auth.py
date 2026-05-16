from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import timedelta
from app.security.auth_backend import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    verify_refresh_token
)
from app.api.dependencies import get_current_user
from app.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    username: str
    role: str
    expires_in: int


@router.post("/token", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    Login with username and password.
    Returns both access token and refresh token.

    - Access token expires in 30 minutes
    - Refresh token expires in 7 days
    """
    user = authenticate_user(
        form_data.username,
        form_data.password
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(
            minutes=settings.access_token_expire_minutes
        )
    )

    # Create refresh token
    refresh_token = create_refresh_token(
        data={"sub": user["username"], "role": user["role"]}
    )

    logger.info(f"Login successful: {user['username']}")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        username=user["username"],
        role=user["role"],
        expires_in=settings.access_token_expire_minutes * 60
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest):
    """
    Get new access token using refresh token.
    Use this when access token expires — no need to login again!
    """
    payload = verify_refresh_token(request.refresh_token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user info
    from app.security.auth_backend import get_user
    user = get_user(payload["username"])

    if not user or not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )

    # Issue new access token
    new_access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(
            minutes=settings.access_token_expire_minutes
        )
    )

    # Issue new refresh token
    new_refresh_token = create_refresh_token(
        data={"sub": user["username"], "role": user["role"]}
    )

    logger.info(f"Token refreshed for: {user['username']}")

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        username=user["username"],
        role=user["role"],
        expires_in=settings.access_token_expire_minutes * 60
    )


@router.get("/me")
async def get_me(
    current_user: dict = Depends(get_current_user)
):
    """Get current user information"""
    return {
        "username": current_user["username"],
        "role": current_user["role"]
    }