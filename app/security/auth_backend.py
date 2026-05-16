from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.config import settings
import logging
import os

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Users loaded from environment — not hardcoded
def _load_users():
    users = {}
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "changeme")

    users[admin_user] = {
        "username": admin_user,
        "hashed_password": pwd_context.hash(admin_pass),
        "is_active": True,
        "role": "admin"
    }
    return users

USERS = _load_users()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def get_user(username: str) -> Optional[dict]:
    return USERS.get(username)

def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user:
        logger.warning(f"User not found: {username}")
        return None
    if not verify_password(password, user["hashed_password"]):
        logger.warning(f"Wrong password for: {username}")
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    logger.info(f"Token created for: {data.get('sub')}")
    return token

def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=7))
    to_encode.update({"exp": expire, "type": "refresh"})
    token = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    logger.info(f"Refresh token created for: {data.get('sub')}")
    return token

def verify_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username = payload.get("sub")
        if username is None:
            return None
        return {"username": username, "role": payload.get("role")}
    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        return None

def verify_refresh_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        if payload.get("type") != "refresh":
            return None
        username = payload.get("sub")
        if username is None:
            return None
        return {"username": username, "role": payload.get("role")}
    except JWTError as e:
        logger.warning(f"Refresh token verification failed: {e}")
        return None