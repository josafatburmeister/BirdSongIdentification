from typing import Union

import torch

from general import logger, PathManager
from training import metrics, metric_logging, model_runner


class ModelEvaluator(model_runner.ModelRunner):
    """
    Evaluates a model on a given dataset.
    """

    def __init__(self, spectrogram_path_manager: PathManager, architecture: str, **kwargs) -> None:
        """

        Args:
            spectrogram_path_manager: PathManager object that manages the directory containing the spectrograms file and
                their labels.
            architecture: Model architecture, either "resnet18", "resnet34", "resnet50", or "densenet121".
        """
        super().__init__(spectrogram_path_manager,
                         architecture, experiment_name="", **kwargs)
        self.datasets = None
        self.dataloaders = None

    def setup_model(self, model: Union[str, torch.nn.Module], device: torch.device,
                    num_classes: int) -> torch.nn.Module:
        """
        Creates model with the required number of classes and the required architecture.

        Args:
            model: Pytorch model to be evaluated.
            device: Pytorch device on which the model evaluation is to be run.
            num_classes: Number of classes in the dataset used to train the model.

        Returns:
            Pytorch model.
        """

        if type(model) == str:
            model_path = model
            pretrained_model = torch.load(model_path, map_location=device)
            model_weights = pretrained_model["state_dict"]
            model = super().setup_model(num_classes)

            model.load_state_dict(model_weights)

        return model

    def evaluate_model(self, model: Union[str, torch.nn.Module], model_name: str, dataset: str) -> None:
        """
        Runs model evaluation.

        Args:
            model: Pytorch model or path to a Pytorch model file.
            model_name: Descriptive name of the model.
            dataset: Name of the dataset to be used for model evaluation.

        Returns:
            None
        """

        # setup datasets
        datasets, dataloaders = self.setup_dataloaders([dataset])
        self.datasets = datasets
        self.dataloaders = dataloaders
        num_classes = self.datasets[dataset].num_classes()

        device = model_runner.ModelRunner.setup_device()
        model = self.setup_model(model, device, num_classes)
        model.to(device)
        model.eval()

        metric_logger = super().setup_metric_logger(
            {"experiment_name": model_name, "split": dataset})

        model_metrics = metrics.Metrics(num_classes=num_classes,
                                        multi_label=self.multi_label_classification)

        for images, labels in self.dataloaders[dataset]:
            images = images.to(device)
            labels = labels.to(device)

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
            metric_logging.TrainingLogger.store_summary_metrics(model_metrics)

        metric_logger.finish()
