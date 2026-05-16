from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.security.auth_backend import verify_token
from app.security.rate_limiter import rate_limiter
from app.models.model_loader import model_loader
import logging

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


async def get_current_user(
    token: str = Depends(oauth2_scheme)
) -> dict:
    
    payload = verify_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


async def check_rate_limit(
    current_user: dict = Depends(get_current_user)
) -> dict:
    
    user_id = current_user["username"]
    allowed, reason = rate_limiter.is_allowed(user_id)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=reason
        )

    return current_user


async def check_model_loaded() -> bool:
    
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet, please try again later"
        )
    return True