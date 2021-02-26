FROM python:3.8.8-slim-buster as base


# shared between builder and runtime image
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        dumb-init \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tf-bodypix


# builder
FROM base as builder

COPY requirements.build.txt ./
RUN pip install --disable-pip-version-check --no-warn-script-location --user -r requirements.build.txt

COPY requirements.txt ./
RUN pip install --disable-pip-version-check --no-warn-script-location --user -r requirements.txt


# runtime image
FROM base

COPY --from=builder /root/.local /root/.local

COPY tf_bodypix ./tf_bodypix

COPY docker/entrypoint.sh ./docker/entrypoint.sh

ENTRYPOINT ["/usr/bin/dumb-init", "--", "/opt/tf-bodypix/docker/entrypoint.sh"]
