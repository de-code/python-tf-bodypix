VENV = venv
PIP = $(VENV)/bin/pip
PYTHON = $(VENV)/bin/python

ARGS =


venv-clean:
	@if [ -d "$(VENV)" ]; then \
		rm -rf "$(VENV)"; \
	fi


venv-create:
	python3 -m venv $(VENV)


dev-install:
	$(PIP) install -r requirements.build.txt
	$(PIP) install -r requirements.dev.txt
	$(PIP) install -r requirements.txt


dev-venv: venv-create dev-install


dev-flake8:
	$(PYTHON) -m flake8 tf_bodypix tests


dev-pylint:
	$(PYTHON) -m pylint tf_bodypix tests


dev-lint: dev-flake8 dev-pylint


dev-pytest:
	$(PYTHON) -m pytest -p no:cacheprovider $(ARGS)


dev-watch:
	$(PYTHON) -m pytest_watch -- -p no:cacheprovider -p no:warnings $(ARGS)


dev-test: dev-lint dev-pytest


run:
	$(PYTHON) -m tf_bodypix $(ARGS)
