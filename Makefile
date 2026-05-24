.PHONY: help install test eval eval-docker site clean

PY      ?= .venv/bin/python
PIP     ?= .venv/bin/pip
MODEL   ?= gpt-4.1
OUT     ?= results/run

help:
	@echo "Targets:"
	@echo "  make install      - create .venv and install requirements"
	@echo "  make test         - run unit tests (no API calls)"
	@echo "  make eval         - run the eval against MODEL (env: OPENAI_API_KEY)"
	@echo "  make eval-docker  - same, inside Docker (no Python install needed)"
	@echo "  make site         - rebuild the static site from results/"
	@echo "  make clean        - remove caches and the .venv"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL=$(MODEL)  (e.g. gpt-4.1, claude-opus-4-7, your-local-model)"
	@echo "  OUT=$(OUT)"
	@echo "  OPENAI_BASE_URL=<optional, for non-OpenAI compatible endpoints>"

.venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install: .venv

test: install
	$(PY) -m pytest tests/ -q

eval: install
	@test -n "$$OPENAI_API_KEY" || (echo "Set OPENAI_API_KEY first" && exit 1)
	$(PY) run_eval.py --model $(MODEL) --out $(OUT)

eval-docker:
	@test -n "$$OPENAI_API_KEY" || (echo "Set OPENAI_API_KEY first" && exit 1)
	OPENAI_API_KEY=$$OPENAI_API_KEY MODEL=$(MODEL) docker compose run --rm eval

site: install
	$(PY) tools/build_site.py
	@echo "Open site/index.html in your browser"

clean:
	rm -rf .venv .pytest_cache __pycache__ */__pycache__ */*/__pycache__
