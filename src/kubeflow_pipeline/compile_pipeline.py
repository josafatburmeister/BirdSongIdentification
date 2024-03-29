import os
import subprocess

import fire
import kfp
from kfp.compiler import compiler
from kubernetes.client import V1Toleration
import yaml

from general import FileManager


def __set_gpu_resources(yaml_file_path: str) -> None:
    """
    Adds GPU resource specifications to a Kubeflow pipeline definition file.

    Args:
        yaml_file_path: Path of a YAML file containing a Kubeflow pipeline definition.

    Returns:
        None
    """

    with open(yaml_file_path, 'r') as stream:
        try:
            pipeline_yaml = yaml.safe_load(stream)
        except yaml.YAMLError:
            raise NameError(f"Could not read pipeline YAML file")

    for idx, template in enumerate(pipeline_yaml["spec"]["templates"]):
        if template["name"] == "train-model":
            pipeline_yaml["spec"]["templates"][idx]["container"]["resources"] = {}
            pipeline_yaml["spec"]["templates"][idx]["container"]["resources"]["limits"] = {}
            pipeline_yaml["spec"]["templates"][idx]["container"]["resources"]["limits"]["nvidia.com/gpu"] = 1

    with open(yaml_file_path, 'w') as stream:
        try:
            yaml.dump(pipeline_yaml, stream, default_flow_style=False)
        except yaml.YAMLError:
            raise NameError(f"Could not write pipeline YAML file")


def __set_docker_registry(input_yaml_file_path: str, output_yaml_file_path: str, docker_registry: str) -> None:
    """
    Adds Docker registry name to pipeline component definition file.

    Args:
        input_yaml_file_path: Path of a YAML file containing a Kubeflow pipeline component definition.
        output_yaml_file_path: Path where the modified Kubeflow pipeline component definition is to be saved.
        docker_registry: Name of the Docker registry from which to pull the Docker image of the pipeline component.

    Returns:
        None
    """

    with open(input_yaml_file_path, 'r') as stream:
        try:
            pipeline_yaml = yaml.safe_load(stream)
        except yaml.YAMLError:
            raise NameError(f"Could not read pipeline YAML file")

    pipeline_yaml["implementation"]["container"]["image"] = FileManager.ensure_trailing_slash(docker_registry) + \
                                                            pipeline_yaml["implementation"]["container"]["image"]

    with open(output_yaml_file_path, 'w') as stream:
        try:
            yaml.dump(pipeline_yaml, stream, default_flow_style=False)
        except yaml.YAMLError:
            raise NameError(f"Could not write pipeline YAML file")


def compile_pipeline(docker_registry: str, use_gpu: bool = True) -> None:
    """
    Compiles full Kubeflow pipeline to a Kubeflow pipeline definition file.

    Args:
        docker_registry: Name of the Docker registry from which to pull the Docker image of the pipeline.
        use_gpu: Whether the pipeline is intended to run on GPU.

    Returns:
        None
    """

    __set_docker_registry('kubeflow_pipeline/download_component.yaml',
                          'kubeflow_pipeline/download_component_temp.yaml',
                          docker_registry)
    __set_docker_registry('kubeflow_pipeline/spectrogram_component.yaml',
                          'kubeflow_pipeline/spectrogram_component_temp.yaml',
                          docker_registry)
    __set_docker_registry('kubeflow_pipeline/training_component.yaml',
                          'kubeflow_pipeline/training_component_temp.yaml',
                          docker_registry)

    download_data_container_op = kfp.components.load_component_from_file(
        os.path.join(os.getcwd(), 'kubeflow_pipeline/download_component_temp.yaml'))

    spectrogram_container_op = kfp.components.load_component_from_file(
        os.path.join(os.getcwd(), 'kubeflow_pipeline/spectrogram_component_temp.yaml'))

    training_container_op = kfp.components.load_component_from_file(
        os.path.join(os.getcwd(), 'kubeflow_pipeline/training_component_temp.yaml'))

    def pipeline(gcs_bucket="bird-song-identification",
                 species_list=None,
                 use_nips4b_species_list=True,
                 maximum_samples_per_class=100,
                 test_size=0.4,
                 min_quality="E",
                 sound_types=None,
                 sexes=None,
                 life_stages=None,
                 early_stopping=False,
                 exclude_special_cases=True,
                 maximum_number_of_background_species=None,
                 maximum_recording_length=None,
                 minimum_change=0.0,
                 monitor="f1-score",
                 clear_audio_cache=False,
                 clear_label_cache=False,
                 chunk_length=1000,
                 include_noise_samples=True,
                 is_hyperparameter_tuning=False,
                 signal_threshold=3,
                 noise_threshold=1,
                 clear_spectrogram_cache=False,
                 architecture="resnet18",
                 batch_size=32,
                 experiment_name="",
                 layers_to_unfreeze=None,
                 learning_rate=0.001,
                 learning_rate_scheduler=None,
                 learning_rate_scheduler_gamma=0.1,
                 learning_rate_scheduler_step_size=7,
                 multi_label_classification=True,
                 multi_label_classification_threshold=0.5,
                 number_epochs=25,
                 number_workers=0,
                 optimizer="Adam",
                 patience=3,
                 track_metrics=True,
                 undersample_noise_samples=True,
                 wandb_entity_name="",
                 wandb_key="",
                 wandb_project_name="",
                 weight_decay=0,
                 verbose_logging=False):
        download_task = download_data_container_op(
            gcs_bucket=gcs_bucket,
            species_list=species_list,
            use_nips4b_species_list=use_nips4b_species_list,
            maximum_samples_per_class=maximum_samples_per_class,
            test_size=test_size,
            min_quality=min_quality,
            sound_types=sound_types,
            sexes=sexes,
            life_stages=life_stages,
            exclude_special_cases=exclude_special_cases,
            maximum_number_of_background_species=maximum_number_of_background_species,
            maximum_recording_length=maximum_recording_length,
            clear_audio_cache=clear_audio_cache,
            clear_label_cache=clear_label_cache,
            verbose_logging=verbose_logging
        )

        spectrogram_task = spectrogram_container_op(
            input_path=download_task.output,
            gcs_bucket=gcs_bucket,
            chunk_length=chunk_length,
            include_noise_samples=include_noise_samples,
            signal_threshold=signal_threshold,
            noise_threshold=noise_threshold,
            clear_spectrogram_cache=clear_spectrogram_cache,
            verbose_logging=verbose_logging
        )

        training_task = training_container_op(
            input_path=spectrogram_task.output,
            gcs_bucket=gcs_bucket,
            include_noise_samples=include_noise_samples,
            is_hyperparameter_tuning=is_hyperparameter_tuning,
            architecture=architecture,
            batch_size=batch_size,
            early_stopping=early_stopping,
            experiment_name=experiment_name,
            layers_to_unfreeze=layers_to_unfreeze,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_gamma=learning_rate_scheduler_gamma,
            learning_rate_scheduler_step_size=learning_rate_scheduler_step_size,
            minimum_change=minimum_change,
            monitor=monitor,
            multi_label_classification=multi_label_classification,
            multi_label_classification_threshold=multi_label_classification_threshold,
            number_epochs=number_epochs,
            number_workers=number_workers,
            optimizer=optimizer,
            patience=patience,
            track_metrics=track_metrics,
            undersample_noise_samples=undersample_noise_samples,
            wandb_entity_name=wandb_entity_name,
            wandb_key=wandb_key,
            wandb_project_name=wandb_project_name,
            weight_decay=weight_decay
        )

        if use_gpu:
            training_task.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-p4')
            gpu_toleration = V1Toleration(effect='NoSchedule', key='gpu', operator='Exists')
            training_task.add_toleration(gpu_toleration)

    pipeline_filename = "birdsong_pipeline.yaml"
    compiler.Compiler().compile(pipeline, pipeline_filename)
    if use_gpu:
        __set_gpu_resources(pipeline_filename)

    os.remove('kubeflow_pipeline/download_component_temp.yaml')
    os.remove('kubeflow_pipeline/spectrogram_component_temp.yaml')
    os.remove('kubeflow_pipeline/training_component_temp.yaml')


def compile_demo_pipeline(docker_registry: str, use_gpu: bool = True) -> None:
    """
    Compiles a demo Kubeflow pipeline (without data download and spectrogram creation stages) to a Kubeflow pipeline
    definition file.

    Args:
        docker_registry: Name of the Docker registry from which to pull the Docker image of the pipeline.
        use_gpu: Whether the pipeline is intended to run on GPU.

    Returns:
        None
    """
    __set_docker_registry('kubeflow_pipeline/demo_data_component.yaml',
                          'kubeflow_pipeline/demo_data_component_temp.yaml',
                          docker_registry)
    __set_docker_registry('kubeflow_pipeline/training_component.yaml',
                          'kubeflow_pipeline/training_component_temp.yaml',
                          docker_registry)

    data_loader_op = kfp.components.load_component_from_file(
        os.path.join(os.getcwd(), 'kubeflow_pipeline/demo_data_component.yaml'))

    training_container_op = kfp.components.load_component_from_file(
        os.path.join(os.getcwd(), 'kubeflow_pipeline/training_component.yaml'))

    def pipeline(gcs_bucket="bird-song-identification",
                 early_stopping=False,
                 minimum_change=0.0,
                 monitor="f1-score",
                 is_hyperparameter_tuning=False,
                 architecture="resnet18",
                 batch_size=32,
                 experiment_name="",
                 layers_to_unfreeze=None,
                 learning_rate=0.001,
                 learning_rate_scheduler=None,
                 learning_rate_scheduler_gamma=0.1,
                 learning_rate_scheduler_step_size=7,
                 multi_label_classification=True,
                 multi_label_classification_threshold=0.5,
                 number_epochs=25,
                 optimizer="Adam",
                 patience=3,
                 undersample_noise_samples=True,
                 weight_decay=0,
                 verbose_logging=False):
        download_task = data_loader_op(
            gcs_bucket=gcs_bucket,
            verbose_logging=verbose_logging
        )

        training_task = training_container_op(
            input_path=download_task.output,
            gcs_bucket=gcs_bucket,
            include_noise_samples=True,
            is_hyperparameter_tuning=is_hyperparameter_tuning,
            architecture=architecture,
            batch_size=batch_size,
            early_stopping=early_stopping,
            experiment_name=experiment_name,
            layers_to_unfreeze=layers_to_unfreeze,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_gamma=learning_rate_scheduler_gamma,
            learning_rate_scheduler_step_size=learning_rate_scheduler_step_size,
            minimum_change=minimum_change,
            monitor=monitor,
            multi_label_classification=multi_label_classification,
            multi_label_classification_threshold=multi_label_classification_threshold,
            number_epochs=number_epochs,
            number_workers=0,
            optimizer=optimizer,
            patience=patience,
            track_metrics=False,
            undersample_noise_samples=undersample_noise_samples,
            wandb_entity_name="",
            wandb_key="",
            wandb_project_name="",
            weight_decay=weight_decay
        )

        if use_gpu:
            training_task.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-p4')
            gpu_toleration = V1Toleration(effect='NoSchedule', key='gpu', operator='Exists')
            training_task.add_toleration(gpu_toleration)

    pipeline_filename = "demo_pipeline.yaml"
    compiler.Compiler().compile(pipeline, pipeline_filename)
    if use_gpu:
        __set_gpu_resources(pipeline_filename)

    os.remove('kubeflow_pipeline/demo_data_component_temp.yaml')
    os.remove('kubeflow_pipeline/training_component_temp.yaml')


def build_docker_image(docker_registry: str, cwd: str = ".", rebuild_prebuild_image: bool = True) -> None:
    """
    Builds and pushes the Docker image used by the Kubeflow pipeline.

    Args:
        docker_registry: Name of the Docker registry to which the Docker image should be pushed.
        cwd: Path of the directory where the Dockerfiles are located.
        rebuild_prebuild_image: Whether the base image (defined by "PrebuildDockerfile") should be rebuild.

    Returns:
        None
    """

    if not os.path.exists(os.path.join(cwd, "Dockerfile")):
        raise NameError(f"Could not find Dockerfile.")
    if not os.path.exists(os.path.join(cwd, "PrebuildDockerfile")):
        raise NameError(f"Could not find PrebuildDockerfile.")

    try:
        if rebuild_prebuild_image:
            subprocess.run(["docker", "build", "-f", "PrebuildDockerfile", "-t", "bird-song-prebuild", "."], cwd=cwd,
                           check=True)
        subprocess.run(["docker", "build", "-t", f"{docker_registry}/bird-song-identification:latest", "--build-arg",
                        "PREBUILD_IMAGE=bird-song-prebuild", "."], cwd=cwd, check=True)
        subprocess.run(["docker", "push", f"{docker_registry}/bird-song-identification:latest"], cwd=cwd, check=True)
    except subprocess.CalledProcessError:
        raise NameError(f"Could not build docker image")


if __name__ == "__main__":
    fire.Fire()
