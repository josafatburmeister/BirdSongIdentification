import json
import torch
from training import training
from general.logging import logger

class TrainingLogger():
    metrics = {
        'f1-score': lambda x: x.f1_score(),
        'precision': lambda x: x.precision(),
        'recall': lambda x: x.recall(),
        'true-positives': lambda x: x.tp,
        'true-negatives': lambda x: x.tn,
        'false-positives': lambda x: x.fp,
        'false-negatives': lambda x: x.fn
    }

    aggregations = {
        'min': lambda x: torch.min(x, dim=2)[0][0],
        'max': lambda x: torch.max(x, dim=2)[0][0],
        'mean': lambda x: torch.mean(x, dim=2)[0]
    }

    def __init__(self, model_trainer, config={}, is_pipeline_run: bool = False):
        self.trainer = model_trainer
        self.config = config
        self.is_pipeline_run = is_pipeline_run


    def log_metrics(self, model_metrics, phase, epoch):
        logger.info("Epoch %d - %s metrics:", epoch+1, phase)

        for metric_name, get_method in self.metrics.items():
            model_metric = get_method(model_metrics)
            log_string = f"    {metric_name}: \t"
            for class_id in range(model_metric.shape[0]):
                if class_id > 0:
                    log_string += ", "
                log_string += "{:.4f} ({})".format(
                    model_metric[class_id].item(), self.trainer.datasets['train'].id_to_class_mapping()[class_id])
            logger.info(log_string)
        logger.info("")


