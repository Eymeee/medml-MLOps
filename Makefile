.PHONY: sync test preprocess train evaluate api

sync:
	uv sync --all-groups

test:
	uv run pytest --cov=src --cov=api --cov-report=term-missing

preprocess:
	uv run python -m src.preprocessing

train:
	uv run python -m src.train

evaluate:
	uv run python -m src.evaluate

api:
	uv run uvicorn api.app:app --reload
