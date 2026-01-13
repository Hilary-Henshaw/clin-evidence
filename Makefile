.PHONY: install install-dev lint format type-check test test-unit \
        test-integration coverage run ingest build docker-build \
        docker-run clean

PYTHON := python3
PIP    := pip
SRC    := src/clinevidence
TESTS  := tests

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

lint:
	ruff check $(SRC) $(TESTS)

format:
	ruff format $(SRC) $(TESTS)

type-check:
	mypy $(SRC)

test:
	pytest $(TESTS) -v

test-unit:
	pytest $(TESTS)/unit -v

test-integration:
	pytest $(TESTS)/integration -v

coverage:
	pytest $(TESTS) --cov=$(SRC) --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

run:
	uvicorn clinevidence.main:app \
		--host 0.0.0.0 --port 8000 --reload

ingest:
	@if [ -z "$(PATH_ARG)" ]; then \
		echo "Usage: make ingest PATH_ARG=./data/raw/doc.pdf"; \
		exit 1; \
	fi
	$(PYTHON) -m clinevidence.scripts.ingest --path $(PATH_ARG)

build:
	$(PIP) install build && $(PYTHON) -m build

docker-build:
	docker build -t clinevidence:latest .

docker-run:
	docker run --rm -p 8000:8000 \
		--env-file .env \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/uploads:/app/uploads \
		clinevidence:latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .mypy_cache .ruff_cache dist build *.egg-info
