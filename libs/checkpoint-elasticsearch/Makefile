.PHONY: test test_watch lint type format

######################
# TESTING AND COVERAGE
######################

start-elasticsearch:
	docker compose -f tests/compose-elasticsearch.yml up -V --force-recreate --wait || ( \
		echo "Failed to start Elasticsearch, printing logs..."; \
		docker compose -f tests/compose-elasticsearch.yml logs; \
		exit 1 \
	)

stop-elasticsearch:
	docker compose -f tests/compose-elasticsearch.yml down

test:
	@make start-elasticsearch
	@uv run pytest $(TEST)
	@EXIT_CODE=$$?; \
	make stop-elasticsearch; \
	exit $$EXIT_CODE

TEST ?= .
test_watch:
	make start-elasticsearch; \
	uv run ptw $(TEST); \
	EXIT_CODE=$$?; \
	make stop-elasticsearch; \
	exit $$EXIT_CODE

######################
# LINTING AND FORMATTING
######################

PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --relative --diff-filter=d main . | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langgraph
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	uv run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE)
	[ "$(PYTHON_FILES)" = "" ] || uv run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

type:
	mkdir -p $(MYPY_CACHE) && uv run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)