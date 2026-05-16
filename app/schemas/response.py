from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(
        ...,
        description="List of embedding vectors"
    )
    shape: List[int] = Field(
        ...,
        description="Shape of embeddings [batch_size, hidden_size]"
    )
    layer_used: int = Field(
        ...,
        description="Which layer was used for extraction"
    )
    model_name: str = Field(
        ...,
        description="Model used for encoding"
    )
    processing_time: float = Field(
        ...,
        description="Time taken in seconds"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "embeddings": [[0.23, -0.87, 0.45]],
                "shape": [1, 768],
                "layer_used": 6,
                "model_name": "distilbert-base-uncased",
                "processing_time": 0.042
            }
        }
    }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_memory_used: Optional[str] = None
    gpu_memory_total: Optional[str] = None
    model_name: str
    version: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int