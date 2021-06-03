ARG PREBUILD_IMAGE
FROM $PREBUILD_IMAGE:latest

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin

RUN pip install -U pip && mkdir -p /app
WORKDIR /app
COPY requirements.txt /app/
RUN pip install pip==20.2.4
RUN pip install -r requirements.txt
COPY . /app
RUN gcloud auth configure-docker --quiet
RUN gcloud auth activate-service-account --key-file="./gcp-bakdata-kubeflow-cluster.json" --quiet
