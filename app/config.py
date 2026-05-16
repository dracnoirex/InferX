from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "InferX"
    app_version: str = "1.0.0"
    debug: bool = False          # ← False by default!
    host: str = "0.0.0.0"
    port: int = 8000

    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    batch_size: int = 8
    device: str = "cuda"

    # No default secret key — must be set via .env!
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    hf_token: Optional[str] = None

    model_config = {
        "env_file": ".env",
        "protected_namespaces": ()
    }

settings = Settings()