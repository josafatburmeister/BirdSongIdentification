from typing import Union

import torch

from general.logging import logger
from training import metrics, model_runner


class ModelEvaluator(model_runner.ModelRunner):
    def __init__(self, spectrogram_path_manager, architecture: str, **kwargs) -> None:
        super().__init__(spectrogram_path_manager, architecture, experiment_name="", **kwargs)
        self.datasets = None
        self.dataloaders = None

    def setup_model(self, model: Union[str, torch.nn.Module], device: torch.device, num_classes: int):
        if type(model) == str:
            model_path = model
            pretrained_model = torch.load(model_path, map_location=device)
            model_weights = pretrained_model["state_dict"]
            model = super().setup_model(num_classes)

            model.load_state_dict(model_weights)

        return model

    def evaluate_model(self, model: Union[str, torch.nn.Module], model_name: str, split: str) -> None:

        # setup datasets
        datasets, dataloaders = self.setup_dataloaders([split])
        self.datasets = datasets
        self.dataloaders = dataloaders
        num_classes = self.datasets[split].num_classes()

        device = model_runner.ModelRunner.setup_device()
        model = self.setup_model(model, device, num_classes)
        model.to(device)
        model.eval()

        metric_logger = super().setup_metric_logger({"experiment_name": model_name, "split": split})

        model_metrics = metrics.Metrics(num_classes=num_classes,
                                        multi_label=self.multi_label_classification)

        for images, labels in self.dataloaders[split]:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                if self.multi_label_classification:
                    predictions = (torch.sigmoid(outputs) > self.multi_label_classification_threshold).int()
                else:
                    _, predictions = torch.max(outputs, 1)

                model_metrics.update(predictions, labels)
        logger.info("Model performance of %s on %s set:", model_name, split)
        metric_logger.log_metrics(model_metrics, "test", 0)
        metric_logger.store_summary_metrics(model_metrics)

        metric_logger.finish()
