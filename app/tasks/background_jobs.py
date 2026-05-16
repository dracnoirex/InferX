from app.tasks.celery_app import celery_app
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def batch_encode_task(self, texts: list, layer: int = None, normalize: bool = True):
    """
    Background task for large batch encoding.
    Submit large jobs asynchronously.
    """
    try:
        from app.inference.batch_processor import batch_processor
        logger.info(f"Background batch task started: {len(texts)} texts")

        results = batch_processor.process(
            texts=texts,
            layer=layer,
            normalize=normalize
        )

        logger.info(f"Background batch task complete: {len(texts)} texts")
        return {
            "status": "complete",
            "total_texts": len(texts),
            "batches_processed": len(results)
        }

    except Exception as exc:
        logger.error(f"Batch task failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task
def cleanup_old_logs():
    """Scheduled task — clean up old request logs"""
    from app.db.database import SessionLocal
    from app.db.models import RequestLog
    from datetime import datetime, timedelta

    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(days=30)
        deleted = db.query(RequestLog).filter(
            RequestLog.created_at < cutoff
        ).delete()
        db.commit()
        logger.info(f"Cleaned up {deleted} old request logs")
        return {"deleted": deleted}
    finally:
        db.close()