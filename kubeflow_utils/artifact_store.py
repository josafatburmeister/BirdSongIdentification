import os
from datetime import datetime
import logging
from kubeflow.metadata import metadata

from kubeflow_utils.config import settings

ARTIFACT_STORE_ENABLED_NAME = "ARTIFACT_STORE_ENABLED"
METADATA_STORE_HOST = settings.artifact_store.host
METADATA_STORE_PORT = settings.artifact_store.port


class ArtifactStore(object):
    def __init__(self):
        if os.environ.get(ARTIFACT_STORE_ENABLED_NAME) is not None:
            self.enabled = os.environ.get(ARTIFACT_STORE_ENABLED_NAME)
        else:
            self.enabled = False

        if self.enabled:
            self.workspace = self.create_workspace()
            self.exec = self.create_execution()

    def re_init(self, pipeline_run: bool = False):
        if self.enabled:
            return

        self.enabled = pipeline_run

        if self.enabled:
            self.workspace = self.create_workspace()
            self.exec = self.create_execution()

    def create_workspace(self):
        logging.info('create workspace')
        return metadata.Workspace(
            store=metadata.Store(grpc_host=METADATA_STORE_HOST, grpc_port=METADATA_STORE_PORT),
            name=settings.artifact_store.workspace.name,
            description=settings.artifact_store.workspace.description)

    def create_execution(self):
        logging.info('create execution')
        r = metadata.Run(
            workspace=self.workspace,
            name="run" + datetime.utcnow().isoformat("T"),
            description=settings.artifact_store.run.description)

        return metadata.Execution(
            name="execution" + datetime.utcnow().isoformat("T"),
            workspace=self.workspace,
            run=r,
            description=settings.artifact_store.execution.description)

    def log_execution_input(self, dataset_name, dataset_description, owner, dataset_path, dataset_version):
        if not self.enabled:
            return

        self.exec.log_input(metadata.DataSet(
            name=dataset_name,
            description=dataset_description,
            owner=owner,
            uri=dataset_path,
            version=dataset_version))
        logging.info("Logged Metadata Dataset")

    def log_execution_output(self, model_name, owner, dataset_path, evaluation):
        if not self.enabled:
            return

        self.exec.log_output(metadata.Metrics(
            name=f'Validation of model: {model_name}',
            owner=owner,
            uri=dataset_path,
            metrics_type=metadata.Metrics.VALIDATION,
            values=evaluation,
        ))
        logging.info("Logged Metadata Metric")

    def log_model(self, model_name, model_version, model_description, owner, gs_path, hyperparameters, model_type,
                  training_framework_name, training_framework_version):
        if not self.enabled:
            return

        self.exec.log_output(metadata.Model(
            name=model_name,
            description=model_description,
            owner=owner,
            uri=gs_path,
            model_type=model_type,
            training_framework={
                "name": training_framework_name,
                "version": training_framework_version
            },
            hyperparameters=hyperparameters,
            version=model_version))
        logging.info("Logged Metadata Model")
