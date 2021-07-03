import logging
import shutil

from data_preparation import filepaths, downloader, spectrograms
from training import training

from general.logging import logger


class PipelineSteps:
    def download_xeno_canto_data(self, gcs_bucket: str, output_path: str, verbose_logging: bool, **kwargs):
        if verbose_logging:
            logger.setLevel(logging.VERBOSE)
        path_manager = filepaths.PathManager(output_path, gcs_bucket=gcs_bucket)
        xc_downloader = downloader.XenoCantoDownloader(path_manager)
        xc_downloader.create_datasets(**kwargs)

    def create_spectrograms(self, input_path: str, gcs_bucket: str, output_path: str, chunk_length: int,
                            clear_spectrogram_cache: bool = False, verbose_logging: bool = False, **kwargs):
        if verbose_logging:
            logger.setLevel(logging.VERBOSE)
        audio_path_manager = filepaths.PathManager(input_path, gcs_bucket=gcs_bucket)
        spectrogram_path_manager = filepaths.PathManager(output_path, gcs_bucket=gcs_bucket)
        spectrogram_creator = spectrograms.SpectrogramCreator(chunk_length, audio_path_manager, spectrogram_path_manager, **kwargs)

        shutil.copy(audio_path_manager.categories_file(), spectrogram_path_manager.categories_file())

        spectrogram_creator.create_spectrograms_for_splits(
            splits=["train", "val", "test"], clear_spectrogram_cache=clear_spectrogram_cache)

        # clean up
        filepaths.PathManager.empty_dir(input_path)

    def train_model(self, input_path: str, gcs_bucket: str, verbose_logging: bool = False, **kwargs):
        if verbose_logging:
            logger.setLevel(logging.VERBOSE)

        spectrogram_path_manager = filepaths.PathManager(input_path, gcs_bucket=gcs_bucket)

        model_trainer = training.ModelTrainer(spectrogram_path_manager, **kwargs)
        model_trainer.train_model()

        # clean up
        filepaths.PathManager.empty_dir(input_path)