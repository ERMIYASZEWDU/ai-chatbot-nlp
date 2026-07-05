# AI Chatbot NLP - Development Makefile
# Run with: make <target>

.PHONY: help install dev install-spacy run test lint format clean docker-build docker-run

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov mypy ruff black

install-spacy: ## Download spaCy language model
	python -m spacy download en_core_web_sm

run: ## Start the development server
	python app.py

test: ## Run all tests with coverage
	pytest --cov=src --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

lint: ## Run linting with ruff
	ruff check src/ app.py

format: ## Format code with black and ruff
	black src/ app.py
	ruff check --fix src/ app.py

typecheck: ## Run type checking with mypy
	mypy src/

check: lint typecheck test ## Run all checks (lint, typecheck, test)

train: ## Train the model
	python train_model.py

clean: ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml

docker-build: ## Build Docker image
	docker build -t ai-chatbot-nlp:latest .

docker-run: ## Run Docker container
	docker run -p 5000:5000 ai-chatbot-nlp:latest

docker-up: ## Run with docker-compose
	docker-compose up -d

docker-down: ## Stop docker-compose services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

.DEFAULT_GOAL := help
