import torch
from transformers import AutoModel, AutoTokenizer
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
        self.is_loaded = False

    def load(self):
        """Load model and tokenizer onto device"""
        try:
            logger.info(f"Loading model: {settings.model_name}")
            logger.info(f"Device: {self.device}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.model_name,
                token=settings.hf_token
            )

            # Load model with hidden states enabled
            self.model = AutoModel.from_pretrained(
                settings.model_name,
                output_hidden_states=True,
                token=settings.hf_token
            )

            # Move model to device
            self.model = self.model.to(self.device)

            # Set to evaluation mode
            self.model.eval()

            self.is_loaded = True
            logger.info(f"✅ Model loaded successfully on {self.device}")

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**2
                logger.info(f"GPU Memory used: {memory_used:.1f} MB")

            # Invalidate cache when new model loads
            try:
                from app.cache.embedding_cache import embedding_cache
                embedding_cache.invalidate_all()
                logger.info("✅ Cache invalidated after model load")
            except Exception as e:
                logger.warning(f"Cache invalidation skipped: {e}")

        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise e

    def unload(self):
        """Unload model and free GPU memory"""
        if self.model is not None:
            # Invalidate cache before unloading
            try:
                from app.cache.embedding_cache import embedding_cache
                embedding_cache.invalidate_all()
                logger.info("✅ Cache invalidated before model unload")
            except Exception as e:
                logger.warning(f"Cache invalidation skipped: {e}")

            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.is_loaded = False
            logger.info("Model unloaded, GPU memory freed")

    def get_model_info(self):
        """Return model information and GPU stats"""
        info = {
            "model_name": settings.model_name,
            "device": str(self.device),
            "is_loaded": self.is_loaded,
        }

        if torch.cuda.is_available():
            info["gpu_memory_used"] = f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f} MB"

        return info


# Global model loader object
model_loader = ModelLoader()