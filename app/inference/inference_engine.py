import time
from typing import List, Optional
from app.inference.layer_extractor import layer_extractor
from app.cache.embedding_cache import embedding_cache
from app.schemas.request import EmbeddingRequest
from app.schemas.response import EmbeddingResponse
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self):
        self.extractor = layer_extractor
        self.cache = embedding_cache

    def run(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Process embedding request with cache support.
        Returns cached result if available, otherwise runs inference.
        """
        start_time = time.time()

        logger.info(f"Processing {len(request.texts)} texts, layer={request.layer}")

        try:
            # Step 1: Check cache
            cached = self.cache.get(
                texts=request.texts,
                layer=request.layer,
                normalize=request.normalize
            )

            if cached:
                processing_time = time.time() - start_time
                logger.info(f"⚡ Cache hit! time={processing_time:.3f}s")

                return EmbeddingResponse(
                    embeddings=cached["embeddings"],
                    shape=cached["shape"],
                    layer_used=cached["layer_used"],
                    model_name=settings.model_name,
                    processing_time=processing_time
                )

            # Step 2: Cache miss - run model inference
            result = self.extractor.extract(
                texts=request.texts,
                layer=request.layer,
                normalize=request.normalize
            )

            # Step 3: Save to cache
            self.cache.set(
                texts=request.texts,
                layer=request.layer,
                normalize=request.normalize,
                result=result
            )

            processing_time = time.time() - start_time

            logger.info(
                f"✅ Done! shape={result['shape']}, "
                f"layer={result['layer_used']}, "
                f"time={processing_time:.3f}s"
            )

            return EmbeddingResponse(
                embeddings=result["embeddings"],
                shape=result["shape"],
                layer_used=result["layer_used"],
                model_name=settings.model_name,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            raise e

    def run_batch(
        self,
        texts: List[str],
        layer: Optional[int] = None,
        normalize: bool = True,
        batch_size: int = None
    ) -> List[EmbeddingResponse]:
        """
        Process large list of texts in smaller batches
        to avoid GPU memory overflow.
        """
        if batch_size is None:
            batch_size = settings.batch_size

        responses = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}, "
                f"texts {i}-{i+len(batch)}/{total}"
            )

            request = EmbeddingRequest(
                texts=batch,
                layer=layer,
                normalize=normalize
            )

            response = self.run(request)
            responses.append(response)

        return responses


# Global inference engine object
inference_engine = InferenceEngine()