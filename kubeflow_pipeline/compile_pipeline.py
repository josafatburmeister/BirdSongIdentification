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

    def pipeline(gcs_bucket="bird-song-identification", species_list=None, use_nips4b_species_list=True,
                 maximum_samples_per_class=100, test_size=0.35,
                 min_quality="E", sound_types=None, sexes=None,
                 life_stages=None, exclude_special_cases=True,
                 maximum_number_of_background_species=None,
                 clear_audio_cache=False, clear_label_cache=False,
                 chunk_length=1000, clear_spectrogram_cache=False, batch_size=32, number_epochs=25, verbose_logging=False):
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
            clear_spectrogram_cache=clear_spectrogram_cache,
            verbose_logging=verbose_logging
        )

        training_task = training_container_op(
            input_path=spectrogram_task.output,
            gcs_bucket=gcs_bucket,
            chunk_length=chunk_length,
            batch_size=batch_size,
            number_epochs=number_epochs
        )

    pipeline_filename = "birdsong_pipeline.zip"
    compiler.Compiler().compile(pipeline, pipeline_filename)
