NOTEBOOK_DIR := notebooks

# Usage:
#   make notebook
#   make notebook TARGET=notebooks/eda.py
#
# TARGET is optional. If omitted, all notebooks in NOTEBOOK_DIR are processed.

TARGET ?=

ifeq ($(TARGET),)
PY_NOTEBOOKS := $(wildcard $(NOTEBOOK_DIR)/*.py)
IPYNB_NOTEBOOKS := $(patsubst %.py,%.ipynb,$(PY_NOTEBOOKS))
else
PY_NOTEBOOKS := $(TARGET)
IPYNB_NOTEBOOKS := $(patsubst %.py,%.ipynb,$(TARGET))
endif

# Kernel to use when executing notebooks
# Cambia esto si tu kernel en Jupyter tiene otro nombre.
KERNEL_NAME ?= proyecto5-grupo5

.PHONY: sync check format run clean notebook

# Format notebook source files without applying fixes.
format:
	@echo "Formatting notebooks..."
	ruff format $(PY_NOTEBOOKS)
	@echo "Done."

# Apply Ruff fixes and format the notebook source files.
check:
	@echo "Checking and fixing notebooks..."
	ruff check --fix --ignore E402 $(PY_NOTEBOOKS)
	ruff format $(PY_NOTEBOOKS)
	@echo "Done."

# Synchronize .py and .ipynb notebooks using Jupytext.
sync:
	@echo "Syncing notebooks..."
	jupytext --sync $(PY_NOTEBOOKS)
	@echo "Done."

# Execute notebooks in place, preserving generated outputs.
run:
	@echo "Executing notebooks..."
	PYTHONPATH=$(CURDIR) uv run jupyter nbconvert --execute \
		--ExecutePreprocessor.kernel_name=$(KERNEL_NAME) \
		--inplace $(IPYNB_NOTEBOOKS)
	@echo "Done."


# Remove notebook metadata while keeping cell outputs.
clean:
	@echo "Cleaning notebook metadata..."
	nbstripout --keep-output $(IPYNB_NOTEBOOKS)
	@echo "Done."

# Full notebook pipeline: lint, sync, execute and clean.
notebook: check sync run clean
	@echo "Notebook pipeline completed successfully."
