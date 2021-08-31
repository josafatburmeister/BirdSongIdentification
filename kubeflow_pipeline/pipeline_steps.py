import logging
import shutil
from typing import List

from downloader import NIPS4BPlusDownloader, XenoCantoDownloader

from spectrograms import SpectrogramCreator
from training import model_evaluator, training, hyperparameter_tuner

from general import logger, PathManager


class PipelineSteps:
    def download_xeno_canto_data(self, gcs_bucket: str, output_path: str, species_list: List[str], verbose_logging: bool, **kwargs):
        if verbose_logging:
            logger.setLevel(logging.VERBOSE)
        path_manager = PathManager(output_path, gcs_bucket=gcs_bucket)
        xc_downloader = XenoCantoDownloader(path_manager)
        xc_downloader.create_datasets(species_list=species_list, **kwargs)

        nips4bplus_downloader = NIPS4BPlusDownloader(path_manager)
        nips4bplus_downloader.download_nips4bplus_dataset(species_list=species_list)

        PathManager.empty_dir(path_manager.cache_dir)

    def create_spectrograms(self, input_path: str, gcs_bucket: str, output_path: str, chunk_length: int, signal_threshold=3, noise_threshold=1,
                            clear_spectrogram_cache: bool = False, verbose_logging: bool = False, **kwargs):
        if verbose_logging:
            logger.setLevel(logging.VERBOSE)
        audio_path_manager = PathManager(input_path, gcs_bucket=gcs_bucket)
        spectrogram_path_manager = PathManager(output_path, gcs_bucket=gcs_bucket)
        spectrogram_creator = SpectrogramCreator(chunk_length, audio_path_manager,
                                                              spectrogram_path_manager, **kwargs)

        shutil.copy(audio_path_manager.categories_file(), spectrogram_path_manager.categories_file())

        spectrogram_creator.create_spectrograms_for_splits(
            splits=["train", "val", "test"], signal_threshold=signal_threshold, noise_threshold=noise_threshold, clear_spectrogram_cache=clear_spectrogram_cache)

        spectrogram_creator.create_spectrograms_for_splits(
            splits=["nips4bplus", "nips4bplus_all"], signal_threshold=0, noise_threshold=0, clear_spectrogram_cache=clear_spectrogram_cache)

        # clean up
        PathManager.empty_dir(input_path)
        PathManager.empty_dir(spectrogram_path_manager.cache_dir)

    def train_model(self, input_path: str, gcs_bucket: str, experiment_name: str = "", verbose_logging: bool = False,
                    **kwargs):
        if verbose_logging:
            logger.setLevel(logging.VERBOSE)

        do_hyperparameter_tuning = False

        for hyperparameter in hyperparameter_tuner.HyperparameterTuner.tunable_parameters():
             if hyperparameter in kwargs and type(kwargs[hyperparameter]) == list:
                do_hyperparameter_tuning = True

        spectrogram_path_manager = PathManager(input_path, gcs_bucket=gcs_bucket)

        if do_hyperparameter_tuning:
            del kwargs["is_hyperparameter_tuning"]
            model_tuner = hyperparameter_tuner.HyperparameterTuner(spectrogram_path_manager=spectrogram_path_manager, experiment_name=experiment_name, **kwargs)
            model_tuner.tune_model()
        else:
            trainer = training.ModelTrainer(spectrogram_path_manager, experiment_name=experiment_name, **kwargs)
            best_average_model, best_minimum_model, best_models_per_class = trainer.train_model()

            evaluator = model_evaluator.ModelEvaluator(spectrogram_path_manager, **kwargs)

            for split in ["test", "nips4bplus", "nips4bplus_all"]:
                evaluator.evaluate_model(model=best_average_model,
                                                     model_name=f"{experiment_name}_best_average_model", split=split)
                evaluator.evaluate_model(model=best_minimum_model,
                                                     model_name=f"{experiment_name}_best_minimum_model", split=split)

                for class_name, model in best_models_per_class.items():
                    evaluator.evaluate_model(model=model,
                                                     model_name=f"{experiment_name}_best_model_{class_name}", split=split)

        # clean up
        PathManager.empty_dir(input_path)
