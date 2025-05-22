# usage: make help

.PHONY: help test format
.DEFAULT_GOAL := help

help: ## this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[0-9a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	echo $(MAKEFILE_LIST)

test: ## run tests
	pytest --forked tests/unit/

format: ## fix formatting
	@if [ ! -d "venv" ]; then \
		python -m venv venv; \
		. venv/bin/activate; \
		pip install pre-commit -U; \
		pre-commit clean; \
		pre-commit uninstall; \
		pre-commit install; \
		deactivate; \
	fi
	. venv/bin/activate && pre-commit run --files $$(git diff --name-only master) && deactivate
