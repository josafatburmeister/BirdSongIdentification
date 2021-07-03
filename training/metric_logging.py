import json
from tabulate import tabulate
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
        logger.info("%s metrics:", phase)
        logger.info("")

        table_headers = ["metric"]

        for class_id, class_name in self.trainer.datasets['train'].id_to_class_mapping().items():
            table_headers.append(class_name)

        metric_rows = []

        for metric_name, get_method in self.metrics.items():
            model_metric = get_method(model_metrics)

            metric_row = [metric_name]
            for class_id in range(model_metric.shape[0]):
                metric_row.append(model_metric[class_id].item())
            metric_rows.append(metric_row)
            if metric_name == "recall":
                metric_rows.append([])

        logger.info(tabulate(metric_rows, headers=table_headers, tablefmt='github', floatfmt=".4f", numalign="center"))
        logger.info("")

    def log_metrics_in_kubeflow(self, avg_model_f1_scores, min_model_f1_scores):
        metrics = {
            'metrics': [
                {
                    'name': "avg_model_min_f1_score",
                    'numberValue': torch.min(avg_model_f1_scores).item(),
                    'format': "RAW",
                },
                {
                    'name': "avg_model_mean_f1_score",
                    'numberValue': torch.mean(avg_model_f1_scores).item(),
                    'format': "RAW",
                },
                {
                    'name': "avg_model_max_f1_score",
                    'numberValue': torch.max(avg_model_f1_scores).item(),
                    'format': "RAW",
                },
                {
                    'name': "min_model_min_f1_score",
                    'numberValue': torch.min(min_model_f1_scores).item(),
                    'format': "RAW",
                },
                {
                    'name': "min_model_mean_f1_score",
                    'numberValue': torch.mean(min_model_f1_scores).item(),
                    'format': "RAW",
                },
                {
                    'name': "min_model_max_f1_score",
                    'numberValue': torch.max(min_model_f1_scores).item(),
                    'format': "RAW",
                }
            ]
        }
        with open("/MLPipeline_Metrics.json", mode="w") as json_file:
            json.dump(metrics, json_file)
