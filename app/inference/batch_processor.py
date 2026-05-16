import time
from typing import List, Optional
from app.schemas.request import EmbeddingRequest
from app.schemas.response import EmbeddingResponse
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Hard limits
MAX_BATCH_SIZE = 100
MAX_TEXT_LENGTH = 10000
MAX_BATCH_TIMEOUT = 30.0  # seconds


class BatchProcessor:
    def __init__(self):
        self.max_batch_size = MAX_BATCH_SIZE
        self.max_text_length = MAX_TEXT_LENGTH
        self.max_timeout = MAX_BATCH_TIMEOUT

    def validate(self, texts: List[str]) -> None:
        """
        Validate batch before processing.
        Raises ValueError if invalid.
        """
        # Check batch size
        if len(texts) == 0:
            raise ValueError("Texts list cannot be empty")

        if len(texts) > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(texts)} exceeds maximum {self.max_batch_size}"
            )

        # Check individual text lengths
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty")

            if len(text) > self.max_text_length:
                raise ValueError(
                    f"Text at index {i} exceeds maximum length "
                    f"({len(text)} > {self.max_text_length})"
                )

    def split_into_batches(
        self,
        texts: List[str],
        batch_size: int = None
    ) -> List[List[str]]:
        """Split large text list into smaller batches"""
        if batch_size is None:
            batch_size = settings.batch_size

        batches = []
        for i in range(0, len(texts), batch_size):
            batches.append(texts[i:i + batch_size])

        logger.info(
            f"Split {len(texts)} texts into "
            f"{len(batches)} batches of size {batch_size}"
        )

        return batches

    def process(
        self,
        texts: List[str],
        layer: Optional[int] = None,
        normalize: bool = True
    ) -> List[EmbeddingResponse]:
        """
        Process texts in batches with timeout protection.
        """
        from app.inference.inference_engine import inference_engine

        # Validate first
        self.validate(texts)

        start_time = time.time()
        responses = []
        batches = self.split_into_batches(texts)

        for i, batch in enumerate(batches):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.max_timeout:
                raise TimeoutError(
                    f"Batch processing timeout after {elapsed:.1f}s. "
                    f"Processed {i}/{len(batches)} batches."
                )

            logger.info(f"Processing batch {i+1}/{len(batches)}")

            request = EmbeddingRequest(
                texts=batch,
                layer=layer,
                normalize=normalize
            )

            response = inference_engine.run(request)
            responses.append(response)

        total_time = time.time() - start_time
        logger.info(
            f"✅ Batch complete! "
            f"{len(texts)} texts in {total_time:.2f}s"
        )

        return responses

    def get_stats(self) -> dict:
        """Return batch processor configuration"""
        return {
            "max_batch_size": self.max_batch_size,
            "max_text_length": self.max_text_length,
            "max_timeout_seconds": self.max_timeout,
            "default_batch_size": settings.batch_size
        }


# Global batch processor
batch_processor = BatchProcessor()