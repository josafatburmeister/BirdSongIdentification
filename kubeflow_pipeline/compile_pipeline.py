import os

import kfp
from kfp.compiler import compiler


def compile_pipeline():
    download_data_container_op = kfp.components.load_component_from_file(
        os.path.join(os.getcwd(), 'kubeflow_pipeline/download_component.yaml'))

    spectrogram_container_op = kfp.components.load_component_from_file(
        os.path.join(os.getcwd(), 'kubeflow_pipeline/spectrogram_component.yaml'))

    training_container_op = kfp.components.load_component_from_file(
        os.path.join(os.getcwd(), 'kubeflow_pipeline/training_component.yaml'))

    def pipeline(gcs_bucket="bird-song-identification",
                 species_list=None,
                 use_nips4b_species_list=True,
                 maximum_samples_per_class=100,
                 test_size=0.35,
                 min_quality="E",
                 sound_types=None,
                 sexes=None,
                 life_stages=None,
                 exclude_special_cases=True,
                 maximum_number_of_background_species=None,
                 clear_audio_cache=False,
                 clear_label_cache=False,
                 chunk_length=1000,
                 include_noise_samples=True,
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
                 track_metrics=True,
                 wandb_entity_name="",
                 wandb_key="",
                 wandb_project_name="",
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
            clear_audio_cache=clear_audio_cache,
            clear_label_cache=clear_label_cache,
            verbose_logging=verbose_logging
        )

        spectrogram_task = spectrogram_container_op(
            input_path=download_task.output,
            gcs_bucket=gcs_bucket,
            chunk_length=chunk_length,
            include_noise_samples=include_noise_samples,
            clear_spectrogram_cache=clear_spectrogram_cache,
            verbose_logging=verbose_logging
        )

        training_task = training_container_op(
            input_path=spectrogram_task.output,
            gcs_bucket=gcs_bucket,
            include_noise_samples=include_noise_samples,
            architecture=architecture,
            batch_size=batch_size,
            chunk_length=chunk_length,
            experiment_name=experiment_name,
            layers_to_unfreeze=layers_to_unfreeze,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_gamma=learning_rate_scheduler_gamma,
            learning_rate_scheduler_step_size=learning_rate_scheduler_step_size,
            multi_label_classification=multi_label_classification,
            multi_label_classification_threshold=multi_label_classification_threshold,
            number_epochs=number_epochs,
            number_workers=number_workers,
            optimizer=optimizer,
            track_metrics=track_metrics,
            wandb_entity_name=wandb_entity_name,
            wandb_key=wandb_key,
            wandb_project_name=wandb_project_name,
        )

    pipeline_filename = "birdsong_pipeline.zip"
    compiler.Compiler().compile(pipeline, pipeline_filename)
