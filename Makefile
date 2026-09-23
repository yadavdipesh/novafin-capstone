# =============================================================================
# novafin-capstone/Makefile
# -----------------------------------------------------------------------------
# One entry point per pipeline stage. Two reasons this exists:
#   1. PYTHONHASHSEED must be exported BEFORE the interpreter starts for full
#      determinism (see utils/seed.py). Every target below does that, so
#      `make train` is reproducible in a way that a bare `python train.py`
#      is not.
#   2. An examiner can reproduce the entire project with `make all` and never
#      read a notebook.
# =============================================================================

PYTHON      ?= python
SEED        ?= 42
PIP         ?= $(PYTHON) -m pip
export PYTHONHASHSEED = $(SEED)

.DEFAULT_GOAL := help
.PHONY: help setup install lint format test test-fast clean clean-artifacts \
        verify fingerprint mlflow-ui leaderboard studies validate-spaces \
        campaign campaign-dry campaigns all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: install  ## Install pinned dependencies and the package in editable mode
	@echo "Environment ready. PYTHONHASHSEED=$(PYTHONHASHSEED)"

install:  ## pip install requirements + editable package
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	$(PIP) install -q -e .

lint:  ## Static checks (ruff)
	$(PYTHON) -m ruff check src tests

format:  ## Auto-fix lint issues and format
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

test:  ## Full pytest suite with coverage
	$(PYTHON) -m pytest --cov=novafin --cov-report=term-missing

test-fast:  ## Skip anything marked slow or needing raw data
	$(PYTHON) -m pytest -m "not slow and not needs_data"

verify:  ## Phase-1 smoke test: config loads, seed pins, theme applies
	$(PYTHON) -c "from novafin.config import load_config; \
	from novafin.utils.seed import seed_everything; \
	c = load_config(); r = seed_everything(c.reproducibility.seed); \
	print('config fingerprint:', c.fingerprint()); \
	print('datasets:', sorted(c.datasets)); \
	print('seed report:', r.as_dict())"

fingerprint:  ## Print the config fingerprint (goes in every report footer)
	@$(PYTHON) -c "from novafin.config import load_config; print(load_config().fingerprint())"

mlflow-ui:  ## Local MLflow UI (no account, no network service required)
	$(PYTHON) -m mlflow ui --backend-store-uri ./mlruns --port 5000

leaderboard:  ## Regenerate reports/tables/leaderboard.csv from MLflow runs
	$(PYTHON) -m novafin.models.leaderboard

studies:  ## List every Optuna study and its progress (use after a reconnect)
	@$(PYTHON) -c "from novafin.config import load_config; import optuna; c = load_config(); [print(f'{s.study_name:<40} {s.n_trials:>4} trials') for s in optuna.get_all_study_summaries(f'sqlite:///{c.paths.optuna_storage}')]"

campaign:  ## Run a Level-4 campaign: make campaign CAMPAIGN=configs/campaigns/<file>.yaml
	@test -n "$(CAMPAIGN)" || (echo "usage: make campaign CAMPAIGN=configs/campaigns/<file>.yaml"; exit 1)
	$(PYTHON) -c "from novafin.models.campaign import run_campaign; r = run_campaign('$(CAMPAIGN)'); print(r.to_frame().to_string(index=False)); print('errors:', r.errors or 'none')"

campaign-dry:  ## Validate a campaign without fitting anything
	@test -n "$(CAMPAIGN)" || (echo "usage: make campaign-dry CAMPAIGN=configs/campaigns/<file>.yaml"; exit 1)
	$(PYTHON) -c "from novafin.models.campaign import run_campaign; r = run_campaign('$(CAMPAIGN)', dry_run=True); print(r.to_frame().to_string(index=False))"

campaigns:  ## List every runnable campaign
	@$(PYTHON) -c "from novafin.models.campaign import list_campaigns; [print(' ', p.name) for p in list_campaigns()]"

validate-spaces:  ## Check every search-space range carries a written rationale
	@$(PYTHON) -c "import yaml, pathlib; from novafin.models.tune import load_search_spaces, validate_search_space; raw = yaml.safe_load(pathlib.Path('configs/search_spaces.yaml').read_text()); mods = [k for k in raw if k != 'defaults']; bad = [p for m in mods for n, s in load_search_spaces(m)[0].items() for p in validate_search_space(s)]; print('search spaces OK' if not bad else 'PROBLEMS: ' + str(bad))"

clean-artifacts:  ## Remove generated models, figures and logs (keeps mlruns)
	rm -rf artifacts/*.pkl artifacts/*.json reports/figures/* reports/logs/*
	@find . -name '.gitkeep' -print0 | xargs -0 -I{} true

clean: clean-artifacts  ## Also remove caches and build output
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

all: install lint test verify  ## Full reproducible run
