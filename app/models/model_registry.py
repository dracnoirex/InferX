from typing import Dict, Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Supported models with metadata
SUPPORTED_MODELS: Dict[str, dict] = {
    "distilbert-base-uncased": {
        "hidden_size": 768,
        "num_layers": 6,
        "max_length": 512,
        "vram_mb": 300,
        "description": "Fast, lightweight BERT variant"
    },
    "bert-base-uncased": {
        "hidden_size": 768,
        "num_layers": 12,
        "max_length": 512,
        "vram_mb": 500,
        "description": "Standard BERT base model"
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "hidden_size": 384,
        "num_layers": 6,
        "max_length": 256,
        "vram_mb": 100,
        "description": "Optimized for sentence embeddings"
    },
    "microsoft/phi-2": {
        "hidden_size": 2560,
        "num_layers": 32,
        "max_length": 2048,
        "vram_mb": 3500,
        "description": "Small but powerful LLM"
    }
}


class ModelRegistry:
    def __init__(self):
        self.current_model = settings.model_name

    def get_model_info(self, model_name: str) -> Optional[dict]:
        """Get metadata for a specific model"""
        return SUPPORTED_MODELS.get(model_name)

    def list_models(self) -> list:
        """List all supported models"""
        models = []
        for name, info in SUPPORTED_MODELS.items():
            models.append({
                "name": name,
                "is_current": name == self.current_model,
                **info
            })
        return models

    def is_supported(self, model_name: str) -> bool:
        """Check if model is supported"""
        return model_name in SUPPORTED_MODELS

    def get_current_model_info(self) -> dict:
        """Get current model metadata"""
        info = self.get_model_info(self.current_model)
        if not info:
            return {"name": self.current_model, "status": "custom model"}
        return {"name": self.current_model, **info}


# Global registry
model_registry = ModelRegistry()