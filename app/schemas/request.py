from pydantic import BaseModel, Field
from typing import List, Optional

class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="List of texts to encode",
        min_length=1,
        max_length=100
    )
    layer: Optional[int] = Field(
        default=None,
        description="Which hidden layer to extract. None = last layer",
        ge=0
    )
    normalize: bool = Field(
        default=True,
        description="Normalize embeddings to unit length"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": ["Hello world", "I love machine learning"],
                "layer": None,
                "normalize": True
            }
        }
    }


class HealthRequest(BaseModel):
    verbose: bool = Field(
        default=False,
        description="Show detailed health info"
    )