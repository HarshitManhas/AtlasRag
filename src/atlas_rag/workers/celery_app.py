"""Celery application and worker configuration."""

from celery import Celery

from atlas_rag.config import get_settings

settings = get_settings()

celery_app = Celery(
    "atlas_rag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["atlas_rag.workers.ingestion_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=600,  # 10 min soft limit
    task_time_limit=900,  # 15 min hard limit
)
