.PHONY: install install-dev test lint format clean docker-build train evaluate help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install with dev dependencies
	pip install -e ".[all]"

test: ## Run tests
	pytest tests/ -v --tb=short

lint: ## Run linter
	ruff check src/ tests/

format: ## Auto-format code
	ruff format src/ tests/

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-build: ## Build Docker image
	docker build -t hf-object-detection .

train: ## Run training (pass CONFIG=path/to/config.yaml)
	python -m src.train --config $(or $(CONFIG),config.yaml)

evaluate: ## Run evaluation (pass CONFIG= and CHECKPOINT=)
	python -m src.evaluate --config $(or $(CONFIG),config.yaml) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT))
