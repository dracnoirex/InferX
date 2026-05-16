from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.schemas.request import EmbeddingRequest
from app.schemas.response import EmbeddingResponse, HealthResponse, ErrorResponse
from app.inference.inference_engine import inference_engine
from app.models.model_loader import model_loader
from app.api.dependencies import check_rate_limit, check_model_loaded
from app.db.database import get_db
from app.db.crud import (
    log_request,
    update_user_activity,
    update_model_stats,
    get_request_stats,
    get_user_stats
)
from app.monitoring.metrics import (
    REQUEST_COUNT,
    INFERENCE_LATENCY,
    update_gpu_metrics,
    track_latency
)
from app.config import settings
import logging
import torch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Inference"])


class InferXException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


@router.post(
    "/encode",
    response_model=EmbeddingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        507: {"model": ErrorResponse, "description": "GPU out of memory"},
    }
)
async def encode(
    request: EmbeddingRequest,
    current_user: dict = Depends(check_rate_limit),
    _: bool = Depends(check_model_loaded),
    db: Session = Depends(get_db)
):
    """
    Extract embeddings from input texts.

    - **texts**: List of sentences to encode (max 100)
    - **layer**: Which hidden layer to extract (None = last layer)
    - **normalize**: Normalize embeddings to unit length
    """
    cache_hit = False

    try:
        # Validate batch size
        if len(request.texts) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many texts. Maximum batch size is 100."
            )

        # Validate text length
        for text in request.texts:
            if len(text) > 10000:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Text too long. Maximum 10,000 characters per text."
                )

        # Run inference
        with track_latency(INFERENCE_LATENCY, ["/encode"]):
            response = inference_engine.run(request)

        # Check if cache hit
        cache_hit = response.processing_time < 0.05

        # Update metrics
        REQUEST_COUNT.labels(endpoint="/encode", status="success").inc()
        update_gpu_metrics()

        # Log to database
        log_request(
            db=db,
            username=current_user["username"],
            endpoint="/encode",
            num_texts=len(request.texts),
            layer_used=response.layer_used,
            processing_time=response.processing_time,
            cache_hit=cache_hit,
            model_name=settings.model_name,
            status_code=200
        )

        # Update user activity
        update_user_activity(db, current_user["username"])

        # Update model stats
        update_model_stats(db, settings.model_name, response.processing_time)

        logger.info(
            f"Encode by {current_user['username']} — "
            f"{len(request.texts)} texts, shape={response.shape}, "
            f"cache_hit={cache_hit}"
        )

        return response

    except HTTPException:
        raise

    except torch.cuda.OutOfMemoryError:
        REQUEST_COUNT.labels(endpoint="/encode", status="oom_error").inc()
        logger.error("GPU out of memory error")
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail="GPU out of memory. Try reducing batch size or text length."
        )

    except RuntimeError as e:
        REQUEST_COUNT.labels(endpoint="/encode", status="runtime_error").inc()
        logger.error(f"Runtime error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference failed. Please try again."
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/encode", status="error").inc()
        logger.error(f"Unexpected error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again."
        )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Check server and model health status"""
    info = model_loader.get_model_info()

    return HealthResponse(
        status="healthy" if info["is_loaded"] else "loading",
        model_loaded=info["is_loaded"],
        device=info["device"],
        gpu_memory_used=info.get("gpu_memory_used"),
        gpu_memory_total=info.get("gpu_memory_total"),
        model_name=settings.model_name,
        version=settings.app_version
    )


@router.get("/layers")
async def get_layers(
    _: bool = Depends(check_model_loaded)
):
    """Get available layers information for the loaded model"""
    total_layers = len(
        model_loader.model.encoder.layer
        if hasattr(model_loader.model, 'encoder')
        else []
    )

    return {
        "model_name": settings.model_name,
        "total_layers": total_layers,
        "recommended_layers": {
            "surface": list(range(0, total_layers // 3)),
            "middle": list(range(total_layers // 3, 2 * total_layers // 3)),
            "deep": list(range(2 * total_layers // 3, total_layers))
        }
    }


@router.get("/stats")
async def get_stats(
    current_user: dict = Depends(check_rate_limit),
    db: Session = Depends(get_db)
):
    """Get API usage statistics"""
    if current_user["role"] == "admin":
        # Admin sees all stats
        stats = get_request_stats(db)
    else:
        # User sees own stats
        stats = get_request_stats(db, current_user["username"])

    return {
        "user": current_user["username"],
        "role": current_user["role"],
        **stats
    }


@router.get("/me/stats")
async def get_my_stats(
    current_user: dict = Depends(check_rate_limit),
    db: Session = Depends(get_db)
):
    """Get current user detailed statistics"""
    return get_user_stats(db, current_user["username"])