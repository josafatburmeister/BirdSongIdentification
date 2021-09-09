from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn

from general import PathManager
from training import dataset, metric_logging
from training.metric_logging import TrainingLogger


class ModelRunner:
    """
    Base class for implementing model training and model evaluation loops.
    """

    @staticmethod
    def setup_device():
        """
        Creates the Pytorch device on which the model training / evaluation is to be run; if a GPU with CUDA support is
        available, it is preferred over the CPU.

        Returns:
            Pytorch device.
        """
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 spectrogram_path_manager: PathManager,
                 architecture: str,
                 experiment_name: str,
                 batch_size: int = 100,
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
            spectrogram_path_manager: PathManager object that manages the directory containing the spectrograms file and
                their labels.
            architecture: Model architecture, either "resnet18", "resnet34", "resnet50", or "densenet121".
            experiment_name: Descriptive name of the experiment.
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
        self.spectrogram_path_manager = spectrogram_path_manager

        self.architecture = architecture
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.include_noise_samples = include_noise_samples
        self.multi_label_classification = multi_label_classification
        self.multi_label_classification_threshold = multi_label_classification_threshold
        self.number_workers = number_workers
        self.track_metrics = track_metrics
        self.undersample_noise_samples = undersample_noise_samples
        self.wandb_entity_name = wandb_entity_name
        self.wandb_key = wandb_key
        self.wandb_project_name = wandb_project_name

        self.is_pipeline_run = self.spectrogram_path_manager.is_pipeline_run

    def setup_metric_logger(self, config=None) -> TrainingLogger:
        if config is None:
            config = {}
        logger = metric_logging.TrainingLogger(self, config, self.is_pipeline_run, track_metrics=self.track_metrics,
                                               wandb_entity_name=self.wandb_entity_name,
                                               wandb_project_name=self.wandb_project_name, wandb_key=self.wandb_key)

        return logger

    def setup_dataloaders(self, dataset_names: list) -> Tuple[
        Dict[str, torch.utils.data.Dataset], Dict[str, torch.utils.data.DataLoader]]:
        """
        Creates datasets and dataloaders for the given dataset names.

        Args:
            dataset_names: List of dataset names (e.g. train, val, or test).

        Returns:
            Datasets and dataloaders for the given datasets names.
        """

        datasets = {}
        dataloaders = {}

        for dataset_name in dataset_names:
            datasets[dataset_name] = dataset.SpectrogramDataset(
                self.spectrogram_path_manager,
                include_noise_samples=self.include_noise_samples, dataset=dataset_name,
                multi_label_classification=self.multi_label_classification,
                undersample_noise_samples=self.undersample_noise_samples)

            shuffle = (dataset_name == "train")
            dataloaders[dataset_name] = DataLoader(
                datasets[dataset_name], batch_size=self.batch_size, sampler=None,
                shuffle=shuffle, num_workers=self.number_workers)

        return datasets, dataloaders

    def setup_model(self, num_classes: int) -> Optional[torch.nn.Module]:
        """
        Creates model with the required number of classes and the required architecture.
        
        Args:
            num_classes: Number of classes in the dataset used to train the model.

        Returns:
            Pytorch model.
        """

        if self.architecture == "resnet18":
            model = models.resnet18(pretrained=True, progress=(not self.is_pipeline_run))
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            model = None

        return model
