VENV = venv
PIP = $(VENV)/bin/pip
PYTHON = $(VENV)/bin/python

VENV_TEMP = venv_temp

ARGS =


IMAGE_URL = https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Person_Of_Interest_-_Panel_%289353656298%29.jpg/640px-Person_Of_Interest_-_Panel_%289353656298%29.jpg
OUTPUT_MASK_PATH = data/example-mask.jpg
OUTPUT_SELECTED_MASK_PATH = data/example-selected-mask.jpg
OUTPUT_COLORED_MASK_PATH = data/example-colored-mask.jpg
OUTPUT_SELECTED_COLORED_MASK_PATH = data/example-selected-colored-mask.jpg
OUTPUT_WEBCAM_MASK_PATH = data/webcam-mask.jpg
MASK_THRESHOLD = 0.75
ADD_OVERLAY_ALPHA = 0.5

SELECTED_PARTS = left_face right_face

WEBCAM_PATH = webcam:0


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


dev-remove-dist:
	rm -rf ./dist


dev-build-dist:
	$(PYTHON) setup.py sdist bdist_wheel


dev-list-dist-contents:
	tar -ztvf dist/tf-bodypix-*.tar.gz


dev-get-version:
	$(PYTHON) setup.py --version


dev-test-install-dist:
	$(MAKE) VENV=$(VENV_TEMP) venv-create
	$(VENV_TEMP)/bin/pip install -r requirements.build.txt
	$(VENV_TEMP)/bin/pip install --force-reinstall ./dist/*.tar.gz
	$(VENV_TEMP)/bin/pip install --force-reinstall ./dist/*.whl


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


convert-example-image-to-selected-mask:
	TF_CPP_MIN_LOG_LEVEL=3 $(PYTHON) -m tf_bodypix \
		image-to-mask \
		--image \
		"$(IMAGE_URL)" \
		--output-mask \
		"$(OUTPUT_SELECTED_MASK_PATH)" \
		--threshold=$(MASK_THRESHOLD) \
		--parts $(SELECTED_PARTS) \
		$(ARGS)


convert-example-image-to-colored-mask:
	TF_CPP_MIN_LOG_LEVEL=3 $(PYTHON) -m tf_bodypix \
		image-to-mask \
		--image \
		"$(IMAGE_URL)" \
		--output-mask \
		"$(OUTPUT_COLORED_MASK_PATH)" \
		--threshold=$(MASK_THRESHOLD) \
		--colored \
		$(ARGS)


convert-example-image-to-selected-colored-mask:
	TF_CPP_MIN_LOG_LEVEL=3 $(PYTHON) -m tf_bodypix \
		image-to-mask \
		--image \
		"$(IMAGE_URL)" \
		--output-mask \
		"$(OUTPUT_SELECTED_COLORED_MASK_PATH)" \
		--threshold=$(MASK_THRESHOLD) \
		--colored \
		--parts $(SELECTED_PARTS) \
		$(ARGS)


webcam:
	TF_CPP_MIN_LOG_LEVEL=3 $(PYTHON) -m tf_bodypix \
		image-to-mask \
		--image \
		"$(WEBCAM_PATH)" \
		--show-output \
		--threshold=$(MASK_THRESHOLD) \
		--add-overlay-alpha=$(ADD_OVERLAY_ALPHA) \
		--colored \
		$(ARGS)
