import torch
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn

from training import dataset, metric_logging


class ModelRunner:
    @staticmethod
    def setup_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 spectrogram_path_manager,
                 architecture: str,
                 chunk_length: int,
                 experiment_name: str,
                 batch_size: int = 100,
                 include_noise_samples: bool = True,
                 multi_label_classification: bool = True,
                 multi_label_classification_threshold: float = 0.5,
                 number_workers=0,
                 track_metrics=True,
                 undersample_noise_samples=True,
                 wandb_entity_name="",
                 wandb_key="",
                 wandb_project_name="",
                 **kwargs
                 ):
        self.spectrogram_path_manager = spectrogram_path_manager

        self.architecture = architecture
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.chunk_length = chunk_length
        self.include_noise_samples = include_noise_samples
        self.multi_label_classification = multi_label_classification
        self.multi_label_classification_threshold = multi_label_classification_threshold
        self.number_workers = number_workers
        self.track_metrics = track_metrics
        self.undersample_noise_samples=undersample_noise_samples
        self.wandb_entity_name = wandb_entity_name
        self.wandb_key = wandb_key
        self.wandb_project_name = wandb_project_name

        self.is_pipeline_run = self.spectrogram_path_manager.is_pipeline_run

    def setup_metric_logger(self, config: dict = {}):
        logger = metric_logging.TrainingLogger(self, config, self.is_pipeline_run, track_metrics=self.track_metrics,
                                                    wandb_entity_name=self.wandb_entity_name,
                                                    wandb_project_name=self.wandb_project_name, wandb_key=self.wandb_key)

        return logger

    def setup_dataloaders(self, splits: list):
        datasets = {}
        dataloaders = {}

        for split in splits:
            datasets[split] = dataset.XenoCantoSpectrograms(
                self.spectrogram_path_manager, chunk_length=self.chunk_length,
                include_noise_samples=self.include_noise_samples, split=split,
                multi_label_classification=self.multi_label_classification, undersample_noise_samples=self.undersample_noise_samples)

            shuffle = (split == "train")
            dataloaders[split] = DataLoader(
                datasets[split], batch_size=self.batch_size, sampler=None,
                shuffle=shuffle, num_workers=self.number_workers)

        return datasets, dataloaders

    def setup_model(self, num_classes: int):
        if self.architecture == "resnet18":
            model = models.resnet18(pretrained=True, progress=(not self.is_pipeline_run))
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            model = None

        return model
