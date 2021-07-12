import torch
from typing import Union

from general.logging import logger
from training import metrics, model_runner

class ModelEvaluator(model_runner.ModelRunner):
    def __init__(self, spectrogram_path_manager, architecture: str, **kwargs):
        super().__init__(spectrogram_path_manager, architecture, experiment_name="", **kwargs)

    def setup_model(self, model):
        if type(model) == str:
            model_path = model
            pretrained_model = torch.load(model_path)
            model_weights = pretrained_model["state_dict"]
            model = super().setup_model()

            model.load_state_dict(model_weights)

        return model

    def evaluate_model_on_test_set(self, model: Union[str, torch.nn.Module], model_name: str):
        self.experiment_name=model_name
        device = model_runner.ModelRunner.setup_device()
        model = self.setup_model(model)
        model.to(device)
        model.eval()

        metric_logger = super().setup_metric_logger()

        model_metrics = metrics.Metrics(num_classes=self.num_classes,
                                        multi_label=self.multi_label_classification)

        for images, labels in self.dataloaders["test"]:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                if self.multi_label_classification:
                    predictions = (torch.sigmoid(outputs) > self.multi_label_classification_threshold).int()
                else:
                    _, predictions = torch.max(outputs, 1)

                model_metrics.update(predictions, labels)
        logger.info("Model performance of %s on evaluation set:", model_name)
        metric_logger.log_metrics(model_metrics, "test", 0)

        metric_logger.finish()
