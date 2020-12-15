FROM python:3.8-buster

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        dumb-init libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tf-bodypix

COPY requirements.build.txt ./
RUN pip install --disable-pip-version-check --user -r requirements.build.txt

COPY requirements.txt ./
RUN pip install --disable-pip-version-check --user -r requirements.txt

COPY tf_bodypix ./tf_bodypix

COPY docker/entrypoint.sh ./docker/entrypoint.sh

ENTRYPOINT ["/usr/bin/dumb-init", "--", "/opt/tf-bodypix/docker/entrypoint.sh"]
