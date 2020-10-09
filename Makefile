VENV = venv
PIP = $(VENV)/bin/pip
PYTHON = $(VENV)/bin/python

ARGS =


IMAGE_URL = https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Person_Of_Interest_-_Panel_%289353656298%29.jpg/640px-Person_Of_Interest_-_Panel_%289353656298%29.jpg
OUTPUT_MASK_PATH = data/example-mask.jpg
MASK_THRESHOLD = 0.75


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
	$(PYTHON) -m flake8 tf_bodypix tests setup.py


dev-pylint:
	$(PYTHON) -m pylint tf_bodypix tests setup.py


dev-lint: dev-flake8 dev-pylint


dev-pytest:
	$(PYTHON) -m pytest -p no:cacheprovider $(ARGS)


dev-watch:
	$(PYTHON) -m pytest_watch -- -p no:cacheprovider -p no:warnings $(ARGS)


dev-test: dev-lint dev-pytest


dev-build-dist:
	$(PYTHON) setup.py sdist


dev-get-version:
	$(PYTHON) setup.py --version


run:
	$(PYTHON) -m tf_bodypix $(ARGS)


convert-example-image-to-mask:
	TF_CPP_MIN_LOG_LEVEL=3 $(PYTHON) -m tf_bodypix \
		image-to-mask \
		--image \
		"$(IMAGE_URL)" \
		--output-mask \
		"$(OUTPUT_MASK_PATH)" \
		--threshold=$(MASK_THRESHOLD) \
		$(ARGS)
