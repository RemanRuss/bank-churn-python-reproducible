.PHONY: setup check-env check-data reproduce test clean

PYTHON ?= python

ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
else
VENV_PYTHON := .venv/bin/python
endif

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo "Environment ready."

check-env:
	@test -f "$(VENV_PYTHON)" || (echo "ERROR: project environment is missing. Run 'make setup' first." && exit 1)

check-data:
	@test -f "data/raw/Bank Customer Churn Prediction.csv" || (echo "ERROR: missing data/raw/Bank Customer Churn Prediction.csv" && echo "See README.md for data download instructions." && exit 1)


reproduce: check-env check-data
	@echo "Reproducing bank-churn analysis with Python..."
	$(VENV_PYTHON) src/pipeline/01_prepare_data.py
	$(VENV_PYTHON) src/pipeline/02_train_models.py
	$(VENV_PYTHON) src/analysis/03_make_results.py
	$(VENV_PYTHON) src/analysis/04_render_report.py
	@echo "Complete. Open results/report/churn_report.html"

test: check-env
	$(VENV_PYTHON) -m pytest

clean:
	rm -f data/processed/churn_clean.csv data/processed/train.csv data/processed/test.csv data/processed/training_medians.json
	rm -f artifacts/models/*.joblib artifacts/models/*.json
	rm -f results/tables/*.csv results/figures/*.png results/report/*.html
	@echo "Generated files removed. Source files are unchanged."
