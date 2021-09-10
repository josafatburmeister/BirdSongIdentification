from typing import Union

import torch

from general import logger, FileManager
from training import metrics, metric_logging, model_runner


class ModelEvaluator(model_runner.ModelRunner):
    """
    Evaluates a model on a given dataset.
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
                 wandb_project_name: str = "") -> None:
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

        super().__init__(spectrogram_file_manager,
                         architecture,
                         batch_size=batch_size,
                         include_noise_samples=include_noise_samples,
                         multi_label_classification=multi_label_classification,
                         multi_label_classification_threshold=multi_label_classification_threshold,
                         number_workers=number_workers,
                         track_metrics=track_metrics,
                         undersample_noise_samples=undersample_noise_samples,
                         wandb_entity_name=wandb_entity_name,
                         wandb_key=wandb_key,
                         wandb_project_name=wandb_project_name)

    def _setup_model(self, model: Union[str, torch.nn.Module],
                    num_classes: int) -> torch.nn.Module:
        """
        Creates model with the required number of classes and the required architecture.

        Args:
            model: Pytorch model to be evaluated.
            num_classes: Number of classes in the dataset used to train the model.

        Returns:
            Pytorch model.
        """

        if type(model) == str:
            model_path = model
            pretrained_model = torch.load(model_path, map_location=self.device)
            model_weights = pretrained_model["state_dict"]
            model = super()._setup_model(num_classes, [])

            model.load_state_dict(model_weights)

        return model

    def evaluate_model(self, model: Union[str, torch.nn.Module], model_name: str, dataset: str) -> None:
        """
        Runs model evaluation.

        Args:
            model: Pytorch model or file_manager to a Pytorch model file.
            model_name: Descriptive name of the model.
            dataset: Name of the dataset to be used for model evaluation.

        Returns:
            None
        """

        # setup datasets
        datasets, dataloaders = self._setup_dataloaders([], [dataset])
        num_classes = datasets[dataset].num_classes()

        model = self._setup_model(model, num_classes)
        model.to(self.device)
        model.eval()

        metric_logger = self._setup_metric_logger(
            {"experiment_name": model_name, "split": dataset})

        model_metrics = metrics.Metrics(num_classes=num_classes,
                                        multi_label=self.multi_label_classification)

        for images, labels in dataloaders[dataset]:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = model(images)
                if self.multi_label_classification:
                    predictions = (torch.sigmoid(
                        outputs) > self.multi_label_classification_threshold).int()
                else:
                    _, predictions = torch.max(outputs, 1)

                model_metrics.update(predictions, labels)
        logger.info("Model performance of %s on %s set:", model_name, dataset)
        metric_logger.log_metrics(model_metrics, "test", 0)
        if self.track_metrics:
            metric_logging.MetricLogger.store_summary_metrics(model_metrics)

        metric_logger.finish()
