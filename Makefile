PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev lint format test check suclaude

install:
	$(PIP) install -r .devcontainer/requirements.txt -e .

install-dev:
	$(PIP) install -r .devcontainer/requirements-dev.txt -e .

lint:
	ruff check .

format:
	ruff format .

test:
	pytest

check: lint
	ruff format --check .
	pytest

claude:
	su - coder -c 'cd $(CURDIR) && claude --dangerously-skip-permissions'
