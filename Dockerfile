# BindsNET with the dependency set pinned in poetry.lock.
#   docker build -t bindsnet .                      # with dev tools (pytest, black, ...)
#   docker build --build-arg DEPS=main -t bindsnet .  # runtime dependencies only
#   docker run --rm -it --gpus all bindsnet bash    # --gpus needs the NVIDIA container toolkit
# The torch wheels from the cu130 index bundle their CUDA libraries, so no CUDA base
# image is needed; the host only needs an NVIDIA driver.
FROM python:3.13-slim

ARG DEPS=dev
ARG POETRY_VERSION=2.4.3

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# libgl1 and libglib2.0-0 are needed by opencv-python.
RUN apt-get update && apt-get install --no-install-recommends -y \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install "poetry==${POETRY_VERSION}"

WORKDIR /bindsnet

# Dependencies first, so editing the code does not re-download them.
COPY pyproject.toml poetry.lock README.md ./
RUN if [ "$DEPS" = "main" ]; then poetry install --no-root --only main; \
    else poetry install --no-root; fi \
    && rm -rf /root/.cache/pypoetry

COPY . .
RUN poetry install --only-root
