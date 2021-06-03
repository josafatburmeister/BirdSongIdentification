import json
import logging
import os
import sys
import typing
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import joblib
from kubeflow import fairing

from kubeflow_utils.artifact_store import ArtifactStore, ARTIFACT_STORE_ENABLED_NAME
from kubeflow_utils.config import settings
from kubeflow_utils.metadata_config import MetadataConfig
from kubeflow_utils.model_storage_utils import gcs_copy, gcs_copy_dir, save_model
from kubeflow_utils.training_result import TrainingResult

GCP_PROJECT = fairing.cloud.gcp.guess_project_name()
# GCS_BUCKET_ID = f'{GCP_PROJECT}-{settings.gcloud.bucket_id}'
GCS_BUCKET_ID = f'{settings.gcloud.bucket_id}'
GCS_BUCKET = f'{settings.gcloud.bucket_prefix}{GCS_BUCKET_ID}'
GCS_BUCKET_PATH = f'{GCS_BUCKET}/{settings.gcloud.bucket_path}'

DOCKER_REGISTRY = f'{settings.docker.registry_prefix}/{GCP_PROJECT}/{settings.docker.folder_path}'
py_version = ".".join([str(x) for x in sys.version_info[0:3]])
use_py_36 = py_version.startswith('3.6')

def get_gcs_model_file(model_file):
    return "{}/models/{}".format(GCS_BUCKET_PATH, model_file)


def get_gcs_data_folder(model_file):
    return "{}/data/{}".format(GCS_BUCKET_PATH, model_file)


class KubeflowServe(ABC):
    def __init__(self):
        self.trained_models = None
        self.artifact_store = ArtifactStore()

    @abstractmethod
    def train_model(self, pipeline_run: bool, **kwargs) -> TrainingResult:
        """

        :rtype: should return TrainingResult
        """
        raise NotImplementedError

    @abstractmethod
    def predict_model(self, models: List[any], **kwargs) -> any:
        """

        :rtype: should return prediction
        """
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> MetadataConfig:
        """

        :rtype: should return model name
        """
        raise NotImplementedError

    def fetch_metadata(self) -> MetadataConfig:
        metadata: MetadataConfig = self.get_metadata()
        if type(metadata) != MetadataConfig:
            raise ValueError(f"{metadata} is not a valid {MetadataConfig} object")
        return metadata

    def train(self, pipeline_run: bool = False, **kwargs):
        self.artifact_store.re_init(pipeline_run)
        logging.info(kwargs)

        model_names = [self.get_metadata().model_names] if type(
            self.get_metadata().model_names) is str else self.get_metadata().model_names
        parsed_model_names = ', '.join(model_names)

        self.artifact_store.log_execution_input(
            owner=self.get_metadata().owner,
            dataset_name=self.get_metadata().dataset_name,
            dataset_description=self.get_metadata().dataset_description,
            dataset_path=get_gcs_data_folder(self.get_metadata().dataset_path),
            dataset_version=self.get_metadata().dataset_version,
        )

        logging.info("Start Training")
        training_result = self.train_model(pipeline_run=pipeline_run, **kwargs)
        logging.info("Finished Training")
        models = training_result.models if isinstance(training_result.models, list) else [training_result.models]

        if pipeline_run:
            metrics = {
                'metrics': [{
                    'name': key,
                    # The name of the metric. Visualized as the column name in the runs table.
                    'numberValue': training_result.evaluation[key],  # The value of the metric. Must be a numeric value.
                    'format': "RAW",
                } for key in training_result.evaluation.keys()]
            }
            with open("/mlpipeline-metrics.json", mode="w") as f:
                json.dump(metrics, f)

        logging.info(
            f'Trained model(s) {model_names} with hyperparameters {training_result.hyperparameters} and evaluation {training_result.evaluation}.')

        self.artifact_store.log_execution_output(
            model_name=parsed_model_names,
            owner=self.get_metadata().owner,
            dataset_path=self.get_metadata().dataset_path,
            evaluation=training_result.evaluation
        )

        gcs_model_paths = []
        for model_file, model in zip(model_names, models):
            # if model is a string it is a path to the saved model
            gcs_model_file = get_gcs_model_file(model_file)

            save_model(model, model_file)

            gcs_copy(model_file, gcs_model_file)
            gcs_model_paths.append(gcs_model_file)

        self.artifact_store.log_model(
            model_version=self.get_metadata().model_version,
            model_name=parsed_model_names,
            model_type=self.get_metadata().model_type,
            model_description=self.get_metadata().model_description,
            owner=self.get_metadata().owner,
            gs_path=', '.join(gcs_model_paths),
            hyperparameters=training_result.hyperparameters,
            training_framework_name=self.get_metadata().training_framework_name,
            training_framework_version=self.get_metadata().training_framework_version,
        )

    def predict(self, features: Dict, feature_names=None, **kwargs) -> any:
        logging.info('predict')
        logging.info('features:')
        for key in features.keys():
            logging.info(f'{key}: {features[key]}')

        model_names = [self.get_metadata().model_names] if type(
            self.get_metadata().model_names) is str else self.get_metadata().model_names
        parsed_model_names = ', '.join(model_names)

        """Download or prepare model files"""
        if not self.trained_models:
            self.trained_models = []
            model_names = [self.get_metadata().model_names] if type(
                self.get_metadata().model_names) is str else self.get_metadata().model_names
            for model_file in model_names:
                if not os.path.isfile(model_file):
                    logging.info(f'Load model {model_file} from gcloud')
                    gcs_model_file = get_gcs_model_file(model_file)
                    gcs_copy(gcs_model_file, model_file)


                model = joblib.load(model_file)
                self.trained_models.append(model)

        logging.info(f"Model(s): {parsed_model_names}")
        result = self.predict_model(models=self.trained_models, **features)

        logging.info(result)

        return result

    @staticmethod
    def __get_message_value(input: Any, output: Any) -> Dict:
        return {
            "input": input,
            "output": output
        }

    def parse_base_image(self):
        model_name = self.get_metadata().model_names if type(self.get_metadata().model_names) is str else \
            self.get_metadata().model_names[0]

        registry = f'{settings.docker.registry_prefix}/{GCP_PROJECT}'
        base = f'{settings.docker.image_name}-{py_version}'
        model_appendix = f'{model_name}-{self.fetch_metadata().model_version}'
        return f'{registry}/{base}-{model_appendix}:latest'

    def get_prebuild_docker_image_name(self):
        if use_py_36:
            return 'model-serve-prebuild-36'
        return 'model-serve-prebuild'

    def build_prebuild_docker_image(self):
        registry = f'{settings.docker.registry_prefix}/{GCP_PROJECT}'
        image_name = self.get_prebuild_docker_image_name()

        os.system(f'docker build --build-arg PY_VERSION={py_version} -t {image_name} -f PrebuildDockerfile .')
        os.system(f'docker tag {image_name}:latest {registry}/{image_name}')
        os.system(f'docker push {registry}/{image_name}')

        logging.info(f"Build prebuild image {registry}/{image_name} on python version {py_version}")

    def build_push_docker_image(self):
        registry = f'{settings.docker.registry_prefix}/{GCP_PROJECT}'
        image = f'{registry}/{self.get_prebuild_docker_image_name()}'

        os.system(f'docker build . --build-arg PREBUILD_IMAGE={image} -t {self.parse_base_image()} -f Dockerfile')
        os.system(f'docker push {self.parse_base_image()}')
        logging.info(f"Build image {self.parse_base_image()} on python version {py_version}")

    def train_online(self):
        logging.info("Train online")

        fairing.config.set_builder('docker', registry=DOCKER_REGISTRY, base_image=self.parse_base_image())
        fairing.config.set_deployer(
            'job',
            namespace=settings.k8s.namespace,
            pod_spec_mutators=[self.artifact_store_pod_spec_mutator])
        create_endpoint = fairing.config.fn(self.__class__)
        create_endpoint()

    def deploy(self):
        logging.info("Deploy model")

        pod_spec_mutators = [self.metadata_pod_spec_mutator]

        fairing.config.set_builder('docker', registry=DOCKER_REGISTRY, base_image=self.parse_base_image())

        fairing.config.set_deployer(
            'serving',
            serving_class=self.__class__.__name__,
            service_type="LoadBalancer",
            namespace=settings.k8s.namespace,
            pod_spec_mutators=pod_spec_mutators)

        # create_endpoint = fairing.config.fn(self.__class__)
        fairing.config.set_preprocessor('function', function_obj=self.__class__)
        preprocessor = fairing.config.get_preprocessor()
        logging.info("Using preprocessor: %s", preprocessor)
        builder = fairing.config.get_builder(preprocessor)
        logging.info("Using builder: %s", builder)
        deployer = fairing.config.get_deployer()

        builder.build()
        pod_spec = builder.generate_pod_spec()
        url = deployer.deploy(pod_spec)

    def artifact_store_pod_spec_mutator(self, backend, pod_spec, namespace):
        pod_spec_env = pod_spec.containers[0].env
        pod_spec_env.append({'name': ARTIFACT_STORE_ENABLED_NAME, 'value': "True"})

    def metadata_pod_spec_mutator(self, backend, pod_spec, namespace):
        pod_spec_env = pod_spec.containers[0].env

        metadata = self.fetch_metadata()

        for metadata_key, metadata_value in vars(metadata).items():
            logging.info(metadata_value)
            if type(metadata_value) is str:
                pod_spec_env.append({'name': f'META_{metadata_key}', 'value': metadata_value})
            else:
                logging.info(",".join(metadata_value))
                pod_spec_env.append({'name': f'META_{metadata_key}', 'value': ",".join(metadata_value)})

        type_hints = typing.get_type_hints(self.predict_model)
        doc_string = self.predict_model.__doc__

        jsonData = {
            'jsonData': {}
        }
        for key in type_hints:
            if key != 'return' and key != 'models':
                jsonData['jsonData'][key] = str(type_hints[key])

        pod_spec_env.append({'name': 'META_REST_CONFIG', 'value': json.dumps(jsonData)})
        pod_spec_env.append({'name': 'META_REST_RETURN_TYPE', 'value': str(type_hints.get('return'))})
        pod_spec_env.append({'name': 'META_REST_DESCRIPTION', 'value': doc_string})

    def download_data(self, source_name: str, destination_folder: str):
        """Downloads a blob from the bucket."""
        path = get_gcs_data_folder(source_name)

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        logging.info(f'Download {path} into {destination_folder}')

        gcs_copy_dir(path, destination_folder)
