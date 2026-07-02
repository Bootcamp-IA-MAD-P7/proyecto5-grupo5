NOTEBOOK_DIR := notebooks

# Usage:
#   make notebook
#   make notebook TARGET=notebooks/eda.py
#
# TARGET is optional. If omitted, all notebooks in NOTEBOOK_DIR are processed.

TARGET ?=

ifeq ($(TARGET),)
PY_NOTEBOOKS := $(NOTEBOOK_DIR)/*.py
IPYNB_NOTEBOOKS := $(NOTEBOOK_DIR)/*.ipynb
else
PY_NOTEBOOKS := $(TARGET)
IPYNB_NOTEBOOKS := $(patsubst %.py,%.ipynb,$(TARGET))
endif

.PHONY: sync check format run clean notebook

# Synchronize .py and .ipynb notebooks using Jupytext.
sync:
	@echo "Syncing notebooks..."
	jupytext --sync $(PY_NOTEBOOKS)
	@echo "Done."

# Apply Ruff fixes and format the notebook source files.
check:
	@echo "Checking and fixing notebooks..."
	ruff check --fix $(PY_NOTEBOOKS)
	ruff format $(PY_NOTEBOOKS)
	@echo "Done."

# Format notebook source files without applying fixes.
format:
	@echo "Formatting notebooks..."
	ruff format $(PY_NOTEBOOKS)
	@echo "Done."

# Execute notebooks in place, preserving generated outputs.
run:
	@echo "Executing notebooks..."
	jupyter nbconvert --execute $(IPYNB_NOTEBOOKS) --inplace
	@echo "Done."

# Remove notebook metadata while keeping cell outputs.
clean:
	@echo "Cleaning notebook metadata..."
	nbstripout --keep-output $(IPYNB_NOTEBOOKS)
	@echo "Done."

# Full notebook pipeline: sync, lint, execute and clean.
notebook: sync check run clean
	@echo "Notebook pipeline completed successfully."