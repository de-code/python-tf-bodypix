VENV = venv

ifeq ($(OS),Windows_NT)
	VENV_BIN = $(VENV)/Scripts
else
	VENV_BIN = $(VENV)/bin
endif

PYTHON = $(VENV_BIN)/python
PIP = $(VENV_BIN)/python -m pip

SYSTEM_PYTHON = python3

VENV_TEMP = venv_temp

ARGS =


IMAGE_URL = https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Person_Of_Interest_-_Panel_%289353656298%29.jpg/640px-Person_Of_Interest_-_Panel_%289353656298%29.jpg
BACKGROUND_IMAGE_URL = https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Gold_Coast_skyline.jpg/640px-Gold_Coast_skyline.jpg
OUTPUT_MASK_PATH = data/example-mask.jpg
OUTPUT_SELECTED_MASK_PATH = data/example-selected-mask.jpg
OUTPUT_COLORED_MASK_PATH = data/example-colored-mask.jpg
OUTPUT_SELECTED_COLORED_MASK_PATH = data/example-selected-colored-mask.jpg
OUTPUT_WEBCAM_MASK_PATH = data/webcam-mask.jpg
MASK_THRESHOLD = 0.75
ADD_OVERLAY_ALPHA = 0.5

SELECTED_PARTS = left_face right_face

WEBCAM_PATH = webcam:0
VIRTUAL_VIDEO_DEVICE = /dev/video2

IMAGE_NAME = de4code/tf-bodypix_unstable
IMAGE_TAG = develop


venv-clean:
	@if [ -d "$(VENV)" ]; then \
		rm -rf "$(VENV)"; \
	fi


venv-create:
	$(SYSTEM_PYTHON) -m venv $(VENV)


dev-install:
	$(PIP) install -r requirements.build.txt
	$(PIP) install -r requirements.dev.txt
	$(PIP) install -r requirements.txt


dev-venv: venv-create dev-install


dev-flake8:
	$(PYTHON) -m flake8 tf_bodypix tests setup.py


dev-pylint:
	$(PYTHON) -m pylint tf_bodypix tests setup.py


dev-mypy:
	$(PYTHON) -m mypy --ignore-missing-imports tf_bodypix tests setup.py


dev-lint: dev-flake8 dev-pylint dev-mypy


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


list-models:
	$(PYTHON) -m tf_bodypix \
		list-models


convert-example-draw-mask:
	$(PYTHON) -m tf_bodypix \
		draw-mask \
		--source \
		"$(IMAGE_URL)" \
		--output \
		"$(OUTPUT_MASK_PATH)" \
		--threshold=$(MASK_THRESHOLD) \
		$(ARGS)


convert-example-draw-selected-mask:
	$(PYTHON) -m tf_bodypix \
		draw-mask \
		--source \
		"$(IMAGE_URL)" \
		--output \
		"$(OUTPUT_SELECTED_MASK_PATH)" \
		--threshold=$(MASK_THRESHOLD) \
		--parts $(SELECTED_PARTS) \
		$(ARGS)


convert-example-draw-colored-mask:
	$(PYTHON) -m tf_bodypix \
		draw-mask \
		--source \
		"$(IMAGE_URL)" \
		--output \
		"$(OUTPUT_COLORED_MASK_PATH)" \
		--threshold=$(MASK_THRESHOLD) \
		--colored \
		$(ARGS)


convert-example-draw-selected-colored-mask:
	$(PYTHON) -m tf_bodypix \
		draw-mask \
		--source \
		"$(IMAGE_URL)" \
		--output \
		"$(OUTPUT_SELECTED_COLORED_MASK_PATH)" \
		--threshold=$(MASK_THRESHOLD) \
		--colored \
		--parts $(SELECTED_PARTS) \
		$(ARGS)


webcam-draw-mask:
	$(PYTHON) -m tf_bodypix \
		draw-mask \
		--source \
		"$(WEBCAM_PATH)" \
		--show-output \
		--threshold=$(MASK_THRESHOLD) \
		--add-overlay-alpha=$(ADD_OVERLAY_ALPHA) \
		$(ARGS)


webcam-blur-background:
	$(PYTHON) -m tf_bodypix \
		blur-background \
		--source \
		"$(WEBCAM_PATH)" \
		--show-output \
		--threshold=$(MASK_THRESHOLD) \
		$(ARGS)


webcam-replace-background:
	$(PYTHON) -m tf_bodypix \
		replace-background \
		--source \
		"$(WEBCAM_PATH)" \
		--background \
		"$(BACKGROUND_IMAGE_URL)" \
		--show-output \
		--threshold=$(MASK_THRESHOLD) \
		$(ARGS)


webcam-v4l2-draw-mask:
	$(PYTHON) -m tf_bodypix \
		draw-mask \
		--source \
		"$(WEBCAM_PATH)" \
		--output=$(VIRTUAL_VIDEO_DEVICE) \
		--threshold=$(MASK_THRESHOLD) \
		--add-overlay-alpha=$(ADD_OVERLAY_ALPHA) \
		$(ARGS)


webcam-v4l2-draw-mask-colored:
	$(PYTHON) -m tf_bodypix \
		draw-mask \
		--source \
		"$(WEBCAM_PATH)" \
		--output=$(VIRTUAL_VIDEO_DEVICE) \
		--threshold=$(MASK_THRESHOLD) \
		--add-overlay-alpha=$(ADD_OVERLAY_ALPHA) \
		--colored \
		$(ARGS)


webcam-v4l2-blur-background:
	$(PYTHON) -m tf_bodypix \
		blur-background \
		--source \
		"$(WEBCAM_PATH)" \
		--output=$(VIRTUAL_VIDEO_DEVICE) \
		--threshold=$(MASK_THRESHOLD) \
		$(ARGS)


webcam-v4l2-replace-background:
	$(PYTHON) -m tf_bodypix \
		replace-background \
		--source \
		"$(WEBCAM_PATH)" \
		--background \
		"$(BACKGROUND_IMAGE_URL)" \
		--output=$(VIRTUAL_VIDEO_DEVICE) \
		--threshold=$(MASK_THRESHOLD) \
		$(ARGS)


docker-build:
	docker build . -t $(IMAGE_NAME):$(IMAGE_TAG)


docker-run:
	docker run \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=unix$$DISPLAY \
		-v /dev/shm:/dev/shm \
		--rm $(IMAGE_NAME):$(IMAGE_TAG) $(ARGS)
