ARG PREBUILD_IMAGE
FROM $PREBUILD_IMAGE:latest

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin

RUN pip install -U pip && mkdir -p /app
WORKDIR /app
COPY requirements.txt /app/
RUN pip install pip==20.2.4
RUN pip install -r requirements.txt

# install libsndfile1 for audio processing
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
    libsndfile1

COPY . /app

# create virtual environment to install our own modules as packages
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV --system-site-packages
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -e .

RUN gcloud auth configure-docker --quiet
RUN gcloud auth activate-service-account --key-file="./gcp-bakdata-kubeflow-cluster.json" --quiet
