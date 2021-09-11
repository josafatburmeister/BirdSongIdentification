from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader

from general import logger, FileManager
from models import model_architectures
from training import dataset, metric_logging
from training.metric_logging import MetricLogger


class ModelRunner:
    """
    Base class for implementing model training and model evaluation loops.
    """

    def __init__(self,
                 spectrogram_file_manager: FileManager,
                 architecture: str,
                 batch_size: int = 128,
                 include_noise_samples: bool = True,
                 multi_label_classification: bool = True,
                 multi_label_classification_threshold: float = 0.5,
                 number_workers: int = 0,
                 track_metrics: bool = True,
                 undersample_noise_samples: bool = True,
                 wandb_entity_name: str = "",
                 wandb_key: str = "",
                 wandb_project_name: str = "",
                 **kwargs
                 ) -> None:
        """

        Args:
            spectrogram_file_manager: FileManager object that manages the directory containing the spectrograms file and
                their labels.
            architecture: Model architecture, either "resnet18", "resnet34", "resnet50", or "densenet121".
            batch_size: Batch size.
            include_noise_samples: Whether spectrograms that are classified as "noise" during noise filtering should be
                included in the spectrogram dataset.
            multi_label_classification: Whether the model should be trained as single-label classification model or as
                multi-label classification model.
            multi_label_classification_threshold: Threshold for assigning samples to positive class in multi-label
                classification, only considered if "multi_label_classification" is set to True.
            number_workers: Number of dataloading workers.
            track_metrics: Whether the model metrics should be logged in Weights and Biases (https://wandb.ai/);
                requires "wandb_entity_name", "wandb_key", and "wandb_project_name" to be set.
            undersample_noise_samples: Whether the number of "noise" spectrograms should be limited to the maximum
                number of spectrograms of a sound class.
            wandb_entity_name: Name of the Weights and Biases account to which the model metrics should be logged; only
                considered if "track_metrics" is set to True.
            wandb_key: API key for the Weights and Biases account to which the model metrics should be logged; only
                considered if "track_metrics" is set to True.
            wandb_project_name: Name of the Weights and Biases project to which the model metrics should be logged; only
                considered if "track_metrics" is set to True.
        """
        self.spectrogram_file_manager = spectrogram_file_manager

        self.architecture = architecture
        self.batch_size = batch_size
        self.include_noise_samples = include_noise_samples
        self.multi_label_classification = multi_label_classification
        self.multi_label_classification_threshold = multi_label_classification_threshold
        self.number_workers = number_workers
        self.track_metrics = track_metrics
        self.undersample_noise_samples = undersample_noise_samples
        self.wandb_entity_name = wandb_entity_name
        self.wandb_key = wandb_key
        self.wandb_project_name = wandb_project_name

        self.is_pipeline_run = self.spectrogram_file_manager.is_pipeline_run
        self.device = self._setup_device()

    def _setup_device(self) -> torch.device:
        """
        Creates the Pytorch device on which the model training is to be run; if a GPU with CUDA support is available,
        it is preferred over the CPU.

        Returns:
            Pytorch device.
        """

        device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info('Device set to: %s', device)
        return torch.device(device)

    def _setup_metric_logger(self, id_to_class_mapping: Dict[int, str], class_to_id_mapping: Dict[str, int],
                             config: Dict[str, Any] = None) -> MetricLogger:
        """
        Creates MetricLogger object.

        Args:
            id_to_class_mapping: Dictionary that maps class indices to human-readable class names.
            class_to_id_mapping: Dictionary that maps human-readable class names to class indices.
            config: The model configuration to be logged, dictionary that maps configuration parameter names to values.

        Returns:
            A MetricLogger object.
        """

        if config is None:
            config = {}
        metric_logger = metric_logging.MetricLogger(id_to_class_mapping,
                                                    class_to_id_mapping,
                                                    config,
                                                    self.is_pipeline_run,
                                                    track_metrics=self.track_metrics,
                                                    wandb_entity_name=self.wandb_entity_name,
                                                    wandb_project_name=self.wandb_project_name,
                                                    wandb_key=self.wandb_key)

        return metric_logger

    def _setup_dataloaders(self, dataset_names_shuffle: List[str], dataset_names_no_shuffle: List[str]) -> Tuple[
        Dict[str, dataset.SpectrogramDataset], Dict[str, torch.utils.data.DataLoader]]:
        """
        Creates datasets and dataloaders for the given dataset names.

        Args:
            dataset_names_shuffle: List of dataset names that should be shuffled (e.g. "train").
            dataset_names_no_shuffle: List of dataset names that should not be shuffled (e.g. "val", or "test").

        Returns:
            Datasets and dataloaders for the given datasets names.
        """

        datasets = {}
        dataloaders = {}

        for dataset_name in dataset_names_shuffle + dataset_names_no_shuffle:
            datasets[dataset_name] = dataset.SpectrogramDataset(
                self.spectrogram_file_manager,
                include_noise_samples=self.include_noise_samples, dataset=dataset_name,
                multi_label_classification=self.multi_label_classification,
                undersample_noise_samples=self.undersample_noise_samples)

            shuffle = (dataset_name in dataset_names_shuffle)
            dataloaders[dataset_name] = DataLoader(
                datasets[dataset_name], batch_size=self.batch_size, sampler=None,
                shuffle=shuffle, num_workers=self.number_workers)

        return datasets, dataloaders

    def _setup_model(self, num_classes: int, layers_to_unfreeze: Union[List[str], str],
                     p_dropout: float = 0) -> torch.nn.Module:
        """
        Creates model with the required number of classes and the required architecture.

        Args:
            num_classes: Number of classes in the dataset used to train the model.
            layers_to_unfreeze: List of model layer names to be unfrozen for fine-tuning; if set to "all", all model
                layers will be fine-tuned.
            p_dropout: Probability of dropout before the fully-connected layer.

        Returns:
            Pytorch model.
        """

        logger.info("Setup %s model: ", self.architecture)
        model = None

        if self.architecture in model_architectures:
            model_architecture = model_architectures[self.architecture]
            model = model_architecture(architecture=self.architecture,
                                       num_classes=num_classes,
                                       layers_to_unfreeze=layers_to_unfreeze,
                                       logger=logger,
                                       p_dropout=p_dropout)

        logger.info("\n")
        return model
