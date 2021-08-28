ARG DEPS=development
ARG NVIDIA_30XX=false

FROM nvidia/cuda:11.1-base AS base-default

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    build-essential libgeos-dev liblzma-dev libssl-dev libbz2-dev curl vim python3.8-dev python-dev git libffi-dev \
       && rm -rf /var/lib/apt/lists/*

# install pyenv
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
RUN echo 'eval "$(pyenv init -)"' >> $HOME/.bashrc

# install python version specified in the .python-version file
COPY .python-version .
RUN PYTHON_VERSION=$(cat .python-version) pyenv install $PYTHON_VERSION && pyenv global $PYTHON_VERSION && pyenv rehash

# install poetry and our package
ENV POETRY_NO_INTERACTION=1\
    # send python output directory to stdout
    PYTHONUNBUFFERED=1\
    PIP_NO_CACHE_DIR=off\
    PIP_DISABLE_PIP_VERSION_CHECK=on\
    PIP_DEFAULT_TIMEOUT=100

# add poetry and stuff to path
#ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH" POETRY_VERSION=1.2.0a2
ENV PATH="$HOME/.local/bin:$PATH"
ENV POETRY_VERSION=1.1.8

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python - &&\
    poetry config virtualenvs.create false

WORKDIR $HOME/bindsnet

COPY pyproject.toml ./

FROM base-default AS base-production
RUN poetry install --no-dev  # this will only install production dependencies

FROM base-default AS base-development
RUN poetry install

FROM base-${DEPS} AS nvidia-30xx-false

# a fix for NVIDIA 30xx GPUs
FROM installed AS nvidia-30xx-true

RUN python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

FROM nvidia-30xx-${NVIDIA_30XX} AS final
