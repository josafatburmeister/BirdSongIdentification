import json
from typing import Any, Dict

import torch
import wandb
from tabulate import tabulate

from general.logging import logger
from training.metrics import Metrics


class TrainingLogger:
    """
    Logs model performance metrics.
    """

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

    def __init__(self, model_trainer, config: Dict[str, Any] = None, is_pipeline_run: bool = False,
                 track_metrics: bool = False, wandb_entity_name: str = "", wandb_key: str = "",
                 wandb_project_name: str = "") -> None:
        """

        Args:
            model_trainer: ModelTrainer object used for model training.
            config: Model training configuration to be logged in Weights and Biases; only
                considered if "track_metrics" is set to True.
            is_pipeline_run: Whether the model training is run in a non-notebook execution environment.
            track_metrics: Whether the model metrics should be logged in Weights and Biases (https://wandb.ai/);
                requires "wandb_entity_name", "wandb_key", and "wandb_project_name" to be set.
            wandb_entity_name: Name of the Weights and Biases account to which the model metrics should be logged; only
                considered if "track_metrics" is set to True.
            wandb_key: API key for the Weights and Biases account to which the model metrics should be logged; only
                considered if "track_metrics" is set to True.
            wandb_project_name: Name of the Weights and Biases project to which the model metrics should be logged; only
                considered if "track_metrics" is set to True.
        """

        if not config:
            config = {}
        self.trainer = model_trainer
        self.config = config
        self.is_pipeline_run = is_pipeline_run
        self.track_metrics = track_metrics

        self.id_to_class_mapping = list(self.trainer.datasets.values())[0].id_to_class_mapping()
        self.class_to_id_mapping = list(self.trainer.datasets.values())[0].class_to_id_mapping()

        if track_metrics:
            wandb.login(key=wandb_key)
            wandb.init(project=wandb_project_name, entity=wandb_entity_name, config=config)
            wandb.run.name = config["experiment_name"]

    def get_run_id(self) -> str:
        """

        Returns:
            Weights and Biases run ID.
        """

        if self.track_metrics:
            return wandb.run.id
        else:
            return ""

    def __print_metrics(self, metrics_object: Metrics) -> None:
        """
        Prints metrics object.

        Args:
            metrics_object: A Metrics object.

        Returns:
            None
        """

        logger.info("")

        table_headers = ["metric"]

        for class_id, class_name in self.id_to_class_mapping.items():
            table_headers.append(class_name)

        metric_rows = []

        for metric_name, get_method in self.metrics.items():
            model_metric = get_method(metrics_object)
            metric_row = [metric_name]
            for class_id in range(model_metric.shape[0]):
                metric_row.append(model_metric[class_id].item())
            metric_rows.append(metric_row)
            if metric_name == "recall":
                metric_rows.append([])

        logger.info(tabulate(metric_rows, headers=table_headers, tablefmt='github', floatfmt=".4f", numalign="center"))
        logger.info("")

    def log_metrics(self, model_metrics: Metrics, phase: str, epoch: int, loss: float = None) -> None:
        """
        Prints metrics object and additionally logs metrics in Weights and Biases if "track_metrics" is True.

        Args:
            model_metrics: A Metrics object.
            phase: Current training phase (e.g. "training" or "validation").
            epoch: Current training epoch.
            loss: Loss of current epoch.

        Returns:
            None
        """

        self.__print_metrics(model_metrics)

        if self.track_metrics:
            tolog = {'phase': phase, 'epoch': epoch, 'loss': loss}
            if loss:
                tolog["loss"] = loss
            for name, get_method in self.metrics.items():
                metric = get_method(model_metrics)
                if metric.shape.numel() == 1:
                    metric_name = name + '_' + phase
                    tolog[metric_name] = metric
                else:
                    for class_id in range(metric.shape.numel()):
                        metric_name = self.id_to_class_mapping[class_id] + '_' + name + '_' + phase
                        tolog[metric_name] = metric[class_id]
            wandb.log(tolog, step=epoch)

    def log_metrics_in_kubeflow(self, best_average_metrics: Metrics, best_minimum_metrics: Metrics) -> None:
        """
        Logs metrics

        Args:
            best_average_metrics: Metrics object representing the performance metrics of the model with the highest
                macro-average F1-score.
            best_minimum_metrics: Metrics object representing the performance metrics of the model with the highest
                minimum class-wise F1-score.

        Returns:
            None
        """
        metrics = {
            'metrics': [
                {
                    'name': "avg_model_min_f1_score",
                    'numberValue': torch.min(best_average_metrics.f1_score()).item(),
                    'format': "RAW",
                },
                {
                    'name': "avg_model_mean_f1_score",
                    'numberValue': torch.mean(best_average_metrics.f1_score()).item(),
                    'format': "RAW",
                },
                {
                    'name': "avg_model_max_f1_score",
                    'numberValue': torch.max(best_average_metrics.f1_score()).item(),
                    'format': "RAW",
                },
                {
                    'name': "min_model_min_f1_score",
                    'numberValue': torch.min(best_minimum_metrics.f1_score()).item(),
                    'format': "RAW",
                },
                {
                    'name': "min_model_mean_f1_score",
                    'numberValue': torch.mean(best_minimum_metrics.f1_score()).item(),
                    'format': "RAW",
                },
                {
                    'name': "min_model_max_f1_score",
                    'numberValue': torch.max(best_minimum_metrics.f1_score()).item(),
                    'format': "RAW",
                }
            ]
        }
        with open("/MLPipeline_Metrics.json", mode="w") as json_file:
            json.dump(metrics, json_file)

    def store_summary_metrics(self, metrics: Metrics) -> None:
        """
        Uploads summary metrics to Weights and Biases.

        Args:
            metrics: A Metrics object.

        Returns:
            None
        """

        wandb.run.summary[f"mean_f1_score_avg_model"] = torch.mean(metrics.f1_score())
        wandb.run.summary[f"min_f1_score_avg_model"] = torch.min(metrics.f1_score())
        wandb.run.summary[f"max_f1_score_avg_model"] = torch.max(metrics.f1_score())

    def print_model_summary(self, best_average_epoch: int, best_average_metrics: Metrics, best_minimum_epoch: int,
                            best_minimum_metrics: Metrics,
                            best_epochs_per_class: Dict[str, int] = None,
                            best_metrics_per_class: Dict[str, Metrics] = None) -> None:
        """
        Prints summary metrics and additionally logs them in Weights and Biases if "track_metrics" is True.

        Args:
            best_average_epoch: Training epoch in which the highest macro-averaged F1-score was achieved.
            best_average_metrics: Metrics object representing the performance metrics of the model with the highest
                macro-average F1-score.
            best_minimum_epoch: Training epoch in which the highest minimum class-wise F1-score was achieved.
            best_minimum_metrics: Metrics object representing the performance metrics of the model with the highest
                minimum class-wise F1-score.
            best_epochs_per_class: Dictionary that maps class names to the epoch in which the highest F1-score for that
                class was achieved.
            best_metrics_per_class: Dictionary that maps class names to the Metrics object of the epoch in which the
                highest F1-score for that class was achieved.

        Returns:
            None
        """

        logger.info("Summary")
        logger.info('-' * 10)
        logger.info("Validation metrics of model with best average F1-Scores (epoch %s):", best_average_epoch)
        self.__print_metrics(best_average_metrics)
        logger.info("Validation metrics of model with best minimum F1-Score (epoch %s):", best_minimum_epoch)
        self.__print_metrics(best_minimum_metrics)

        if self.track_metrics:
            self.store_summary_metrics(best_average_metrics)
        if best_metrics_per_class:
            for class_name, best_class_metrics in best_metrics_per_class.items():
                logger.info("Validation metrics of best model for class %s (epoch %s):", class_name,
                            best_epochs_per_class[class_name])
                self.__print_metrics(best_class_metrics)

                if self.track_metrics:
                    for metric_name, get_method in self.metrics.items():
                        metric = get_method(best_class_metrics)
                        wandb.run.summary[f"{class_name}_{metric_name}_best_model_val"] = metric[
                            self.class_to_id_mapping[class_name]].item()
                        wandb.run.summary[f"{class_name}_{metric_name}_best_epoch_val"] = best_epochs_per_class[
                            class_name]

        if self.track_metrics:
            for metric_name, get_method in self.metrics.items():
                avg_metric = get_method(best_average_metrics)
                min_metric = get_method(best_minimum_metrics)
                for class_id, class_name in self.id_to_class_mapping.items():
                    wandb.run.summary[f"{class_name}_{metric_name}_avg_model"] = avg_metric[class_id].item()
                    wandb.run.summary[f"{class_name}_best_epoch_avg_model"] = best_average_epoch
                    wandb.run.summary[f"{class_name}_{metric_name}_min_model"] = min_metric[class_id].item()
                    wandb.run.summary[f"{class_name}_best_epoch_min_model"] = best_minimum_epoch

    def finish(self) -> None:
        """
        Finishes Weights and Biases run.

        Returns:
            None
        """
        if self.track_metrics:
            wandb.run.finish()
