.PHONY: dev test lint format migrate migration docker-up docker-down worker seed eval install

install:
	pip install -e ".[dev]"

dev:
	uvicorn atlas_rag.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=atlas_rag --cov-report=html

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

migrate:
	alembic upgrade head

migration:
	alembic revision --autogenerate -m "$(msg)"

docker-up:
	docker compose up -d

docker-down:
	docker compose down

worker:
	celery -A atlas_rag.workers.celery_app worker --loglevel=info

seed:
	python scripts/seed_data.py

eval:
	python scripts/run_evaluation.py
