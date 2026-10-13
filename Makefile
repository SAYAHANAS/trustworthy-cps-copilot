PY ?= python3
RUN = PYTHONPATH=src $(PY) -m cpscopilot.cli

.PHONY: install test test-fast lint smoke synthetic-seeds fault-injection figures

install:            ## editable install with dev + LLM extras
	$(PY) -m pip install -e ".[dev,llm]"

test:               ## full suite incl. end-to-end runs (~2 min)
	$(PY) -m pytest

test-fast:          ## unit tests only
	$(PY) -m pytest -m "not slow"

lint:
	ruff check src tests scripts

smoke:              ## one quick end-to-end experiment
	$(RUN) run --config configs/experiments/synthetic_smoke.yaml

synthetic-seeds:    ## full synthetic study over five seeds (+ aggregate)
	$(RUN) run --config configs/experiments/synthetic_full.yaml --seeds 0 1 2 3 4

fault-injection:    ## hallucination-audit sensitivity experiment
	$(RUN) run --config configs/experiments/synthetic_fault_injection.yaml

figures:            ## make figures RUNS="runs/<id> runs/<id>"
	PYTHONPATH=src $(PY) scripts/generate_paper_figures.py $(RUNS) --formats pdf svg pgf
