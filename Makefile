.PHONY: install-dev install-hooks install-dotnet-tools install-python-tools format lint

install-dev: install-hooks install-dotnet-tools install-python-tools

install-hooks:
	pip install pre-commit
	pre-commit install

install-dotnet-tools:
	dotnet tool install -g dotnet-format

install-python-tools:
	pip install black flake8 isort pylint

format:
	dotnet format
	black .
	isort .
	prettier --write "**/*.{json,yaml,yml}"

lint:
	dotnet format --verify-no-changes
	black --check .
	isort --check .
	flake8
	pylint **/*.py
