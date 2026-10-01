# POSIX shell. On Windows without make, run the commands of the README directly.
PYTHON ?= python
export PYTHONPATH := backend

.PHONY: install lint format test eval eval-hybrid ingest api front up down

install:
	$(PYTHON) -m pip install -r backend/requirements-dev.txt

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest -q

# Offline evaluation gate, as run in CI (no model download, no LLM).
eval:
	RETRIEVAL_MODE=bm25 LLM_ENABLED=off $(PYTHON) -m app.eval.run_eval

# Same evaluation with the embedding model (downloads it on first use).
eval-hybrid:
	RETRIEVAL_MODE=hybrid LLM_ENABLED=off $(PYTHON) -m app.eval.run_eval

ingest:
	$(PYTHON) -c "from app.ingestion.ingest import ingest_docs; print(ingest_docs().model_dump_json(indent=2))"

api:
	$(PYTHON) -m uvicorn app.main:app --app-dir backend --port 8000

front:
	$(PYTHON) -m streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false

up:
	docker compose up --build -d

down:
	docker compose down
