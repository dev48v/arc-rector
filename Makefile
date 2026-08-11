.DEFAULT_GOAL := help
PY ?= python

.PHONY: help up down logs status install demo demo-offline ingest ask eval swap test levels doctor ui ui-docker clean nuke

UI_HOST ?= 127.0.0.1
UI_PORT ?= 8800

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## -- stack ------------------------------------------------------------------
up:  ## Start Qdrant + Langfuse
	docker compose up -d
	@echo "Qdrant   http://localhost:6333/dashboard"
	@echo "Langfuse http://localhost:3000  (demo@arc-rector.local / arcrector123)"

up-lite:  ## Start Qdrant only (low-RAM: no Langfuse stack)
	docker compose up -d qdrant
	@echo "Run with ARC_L1_OBSERVABILITY=none"

down:  ## Stop the stack, keep data
	docker compose down

logs:  ## Follow all container logs
	docker compose logs -f

status:  ## Show container status
	docker compose ps

## -- setup ------------------------------------------------------------------
install:  ## Install the package with every runnable extra
	$(PY) -m pip install -e ".[all,dev]"

models:  ## Pull the default Ollama models
	ollama pull nomic-embed-text
	ollama pull llama3.1:8b

## -- run --------------------------------------------------------------------
demo:  ## Full end-to-end demo against the real stack
	$(PY) -m arc_rector.demo

demo-offline:  ## Same 5 checks with no containers, no model, no network
	$(PY) -m arc_rector.demo --offline

ingest:  ## Parse, chunk, embed and store the corpus
	$(PY) -m arc_rector.cli ingest --reset

ask:  ## Ask one question: make ask Q="..."
	@$(PY) -m arc_rector.cli ask "$(or $(Q),Which vector database is the default?)"

ui:  ## Serve the web chat UI on http://127.0.0.1:8800
	$(PY) -m arc_rector.server --host $(UI_HOST) --port $(UI_PORT)

ui-docker:  ## Same UI, in a container against the compose stack
	docker compose up -d --build ui
	@echo "UI http://localhost:$(UI_PORT)"

eval:  ## Run the Ragas eval harness
	$(PY) -m arc_rector.eval_harness

swap:  ## One question across 2 vector DBs x 2 frameworks
	$(PY) -m arc_rector.swap_demo

levels:  ## List every adapter registered for every level
	$(PY) -m arc_rector.cli levels

doctor:  ## Probe every selected level and report health
	$(PY) -m arc_rector.cli doctor

## -- develop ----------------------------------------------------------------
test:  ## Run the offline test suite
	$(PY) -m pytest -q

test-cov:  ## Test suite with coverage
	$(PY) -m pytest --cov=arc_rector --cov-report=term-missing

clean:  ## Remove caches and local runtime state
	rm -rf .pytest_cache .ruff_cache .arc_rector htmlcov .coverage
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true

nuke:  ## Stop the stack and DELETE all volumes
	docker compose down -v
