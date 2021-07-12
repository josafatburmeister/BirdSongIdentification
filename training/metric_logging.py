import json
from tabulate import tabulate
import torch
import wandb

from general.logging import logger


class TrainingLogger:
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

    def __init__(self, model_trainer, config=None, is_pipeline_run: bool = False, track_metrics=False,
                 wandb_entity_name="", wandb_project_name="", wandb_key=""):
        if not config:
            config = {}
        self.trainer = model_trainer
        self.config = config
        self.is_pipeline_run = is_pipeline_run
        self.track_metrics = track_metrics

        if track_metrics:
            wandb.login(key=wandb_key)
            wandb.init(project=wandb_project_name, entity=wandb_entity_name, config=config)
            wandb.run.name = config["experiment_name"]

    def get_run_id(self):
        if self.track_metrics:
            return wandb.run.id
        else:
            return ""

    def print_metrics(self, metrics_object):
        logger.info("")

        table_headers = ["metric"]

        for class_id, class_name in self.trainer.datasets['train'].id_to_class_mapping().items():
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

    def log_metrics(self, model_metrics, phase, epoch, loss=None):
        self.print_metrics(model_metrics)

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
                        metric_name = self.trainer.datasets['train'].id_to_class_name(
                            class_id) + '_' + name + '_' + phase
                        tolog[metric_name] = metric[class_id]
            wandb.log(tolog, step=epoch)

    def log_metrics_in_kubeflow(self, best_average_metrics, best_minimum_metrics, best_metrics_per_class=None):
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

    def print_model_summary(self, best_average_epoch, best_average_metrics, best_minimum_epoch, best_minimum_metrics,
                            best_epochs_per_class=None, best_metrics_per_class=None):
        logger.info("Summary")
        logger.info('-' * 10)
        logger.info("Validation metrics of model with best average F1-Scores (epoch %s):", best_average_epoch)
        self.print_metrics(best_average_metrics)
        logger.info("Validation metrics of model with best minimum F1-Score (epoch %s):", best_minimum_epoch)
        self.print_metrics(best_minimum_metrics)

        if best_metrics_per_class:
            for class_name, best_class_metrics in best_metrics_per_class.items():
                logger.info("Validation metrics of best model for class %s (epoch %s):", class_name,
                            best_epochs_per_class[class_name])
                self.print_metrics(best_class_metrics)

                if self.track_metrics:
                    for metric_name, get_method in self.metrics.items():
                        metric = get_method(best_class_metrics)
                        wandb.run.summary[f"{class_name}_{metric_name}_best_model_val"] = metric[
                            self.trainer.datasets["train"].class_name_to_id(class_name)].item()
                        wandb.run.summary[f"{class_name}_{metric_name}_best_epoch_val"] = best_epochs_per_class[
                            class_name]

        if self.track_metrics:
            for metric_name, get_method in self.metrics.items():
                avg_metric = get_method(best_average_metrics)
                min_metric = get_method(best_minimum_metrics)
                for class_id, class_name in self.trainer.datasets['train'].id_to_class_mapping().items():
                    wandb.run.summary[f"{class_name}_{metric_name}_avg_model"] = avg_metric[class_id].item()
                    wandb.run.summary[f"{class_name}_best_epoch_avg_model"] = best_average_epoch
                    wandb.run.summary[f"{class_name}_{metric_name}_min_model"] = min_metric[class_id].item()
                    wandb.run.summary[f"{class_name}_best_epoch_min_model"] = best_minimum_epoch

    def finish(self):
        if self.track_metrics:
            wandb.run.finish()
