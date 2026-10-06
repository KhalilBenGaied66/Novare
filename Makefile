# POSIX shell. On Windows without make, run the commands of the README directly.
PYTHON ?= python
export PYTHONPATH := backend

.PHONY: install lint format test eval eval-hybrid eval-llm ingest api front up down

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

# Same evaluation with models writing the answers and judging them: two open-weight
# models served by a local Ollama, no key and no cost. Measured, not gated.
eval-llm:
	RETRIEVAL_MODE=hybrid LLM_ENABLED=on LLM_TIMEOUT_S=240 \
	RAG_MODEL=ollama_chat/qwen3.5:4b AGENT_MODEL=ollama_chat/qwen3.5:9b \
	JUDGE_MODEL=ollama_chat/qwen3.5:9b \
	LLM_EXTRA_PARAMS='{"num_ctx": 16384, "temperature": 0, "reasoning_effort": "none"}' \
	$(PYTHON) -m app.eval.run_eval --judge --no-gate

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
