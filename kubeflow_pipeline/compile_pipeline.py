import os

import kfp
from kfp.compiler import compiler


def compile_pipeline():
    download_data_container_op = kfp.components.load_component_from_file(
        os.path.join(os.getcwd(), 'kubeflow_pipeline/download_component.yaml'))

    spectrogram_container_op = kfp.components.load_component_from_file(
        os.path.join(os.getcwd(), 'kubeflow_pipeline/spectrogram_component.yaml'))

    def pipeline(species_list, data_dir="bird_song_identification", maximum_samples_per_class=100, test_size=0.35, min_quality="E", sound_types=None, sexes=None,
                 life_stages=None, exclude_special_cases=True, maximum_number_of_background_species=None, chunk_length=1000, verbose=False):
        download_task = download_data_container_op(
            data_dir=data_dir,
            species_list=species_list,
            maximum_samples_per_class=maximum_samples_per_class,
            test_size=test_size,
            min_quality=min_quality,
            sound_types=sound_types,
            sexes=sexes,
            life_stages=life_stages,
            exclude_special_cases=exclude_special_cases,
            maximum_number_of_background_species=maximum_number_of_background_species
        )

        spectrogram_task = spectrogram_container_op(
            input_path=download_task.output,
            data_dir=data_dir,
            chunk_length=chunk_length
        )

    pipeline_filename = "birdsong_pipeline.zip"
    compiler.Compiler().compile(pipeline, pipeline_filename)
